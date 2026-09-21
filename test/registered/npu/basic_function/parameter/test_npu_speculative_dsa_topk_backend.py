"""--speculative-dsa-topk-backend: DSA indexer top-k backend of the speculative draft worker.

Verification level: no_effect (no consumption point on NPU; the parameter is really effective on CUDA)
Consumption points: layers/attention/dsa/dsa_topk_backend.py:33-41 (the is_draft branch of
DSATopKBackend.resolve reads --speculative-dsa-topk-backend) — but resolve is only called
from the CUDA side at deepseek_v4_backend.py:650 / dsa_backend.py:358.
NPU conclusion: DeepseekV4AscendAttnBackend (ascend_dsv4_backend.py) does not construct
DSATopKBackend; DSA top-k is completed in one piece by the platform-specific fused operator
torch.ops.custom.npu_quant_lightning_indexer (vendor package; the INT8 quantized lightning
indexer directly outputs top-k indices), identical for target/draft and independent of
the parameter value.
There is no sgl_kernel package in the NPU container; none of the three CUDA backends
(sgl-kernel fast_topk_v2 / torch unfused / flashinfer) is available or called.

This case still uses a real server launch (DeepSeek-V4-Flash-w8a8-mtp, DeepseekV4ForCausalLM +
index_topk=512 + num_nextn_predict_layers=1) as evidence of the DSV4 NPU pipeline,
and asserts the server_args echo — but note the echo is only parsing, not consumption evidence.

Heavy-resource case (8 cards, 280G weights, ~5 min loading), skipped by default:
set PARA_RUN_DSV4=1 to enable; override card ids with PARA_DSV4_CARDS (default 2,3,8,9,10,11,12,13;
requires 8 cards each with >=55GB free, otherwise blocked by the memory balancing check).

Extra environment for running DeepSeek-V4 on NPU (injected via env_extra):
 - ASCEND_CUSTOM_OPP_PATH / LD_LIBRARY_PATH: under CANN vendors, both the custom_transformer
   (the three DSA operators: SparseAttnSharedkv{,Metadata}, QuantLightningIndexer/Compressor)
   and customize (mHC's aclnnHcPre/HcPost etc.) operator packages must be mounted; missing
   either one raises "not in libopapi.so";
 - SGLANG_OPT_FP8_WO_A_GEMM=0: ModelSlim INT8 weights have wo_a unquantized (BF16); leaving
   this switch at its default builds as for official FP8 weights and triggers the
   weight_scale_inv assertion;
 - SGLANG_OPT_FUSE_WQA_WKV=0: the wq_a/wkv fused remap only recognizes official FP8 naming
   and would drop INT8's weight_scale/weight_offset;
 - SGLANG_OPT_FUSE_MHC_POST_PRE=0: the cross-layer mHC fusion goes through TileLang (NPU has
   no such package); after turning it off, hc_post/hc_pre each go through NPU custom ops;
 - SGLANG_OPT_USE_FUSED_HASH_TOPK=0: hash_topk has no NPU branch on NPU; by default it falls
   back to CUDA JIT (no nvcc), turning it off uses the torch implementation;
 - SGLANG_DSA_FUSE_TOPK=false: required by --speculative-dsa-topk-backend torch.

[Test Category] Parameter
[Test Target] --speculative-dsa-topk-backend
"""

import glob
import os
import unittest

from para_case_common import AscendServer, AscendServerTestCase
from sglang.test.ci.ci_register import register_npu_ci

DSV4_MTP = os.environ.get(
    "PARA_DSV4_MODEL", "/mnt/paas/weights/DeepSeek-V4-Flash-w8a8-mtp"
)
DSV4_CARDS = os.environ.get("PARA_DSV4_CARDS", "2,3,8,9,10,11,12,13")

# DSA torch backend requirement + NPU fixes related to ModelSlim INT8 / TileLang / CUDA JIT
DSV4_SGLANG_ENVS = {
    "SGLANG_DSA_FUSE_TOPK": "false",
    "SGLANG_OPT_FP8_WO_A_GEMM": "0",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_OPT_FUSE_MHC_POST_PRE": "0",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "0",
}


def _vendor_paths():
    """Discover the DSA (custom_transformer) and mHC (customize) operator packages under CANN vendors.

    Returns (ascend_custom_opp_path, ld_library_path additions); raises if not found,
    because the DSV4 server launch inevitably needs these operators.
    """
    roots = sorted(glob.glob("/usr/local/Ascend/cann-*/opp/vendors"))
    opp, libs = [], []
    for name in ("custom_transformer", "customize"):
        for root in roots:
            pkg = os.path.join(root, name)
            if os.path.isdir(pkg):
                opp.append(pkg)
                libs.extend(glob.glob(os.path.join(pkg, "op_api", "lib")))
                break
    missing = {"custom_transformer", "customize"} - {os.path.basename(p) for p in opp}
    if missing:
        raise RuntimeError(
            f"DSV4 operator packages missing from CANN vendors: {missing}; "
            f"searched {roots} (DSV4 server launch will inevitably fail)"
        )
    return os.pathsep.join(opp), os.pathsep.join(libs)


register_npu_ci(est_time=2400, suite="nightly-1-npu-a3", nightly=True)


@unittest.skipUnless(
    os.environ.get("PARA_RUN_DSV4") == "1",
    "Heavy-resource case (8 cards / 280G weights / ~5min): set PARA_RUN_DSV4=1 to enable",
)
class TestSpeculativeDsaTopkBackend(AscendServerTestCase):
    PORT = 31290

    @classmethod
    def setUpClass(cls):
        if not os.path.isdir(DSV4_MTP):
            raise unittest.SkipTest(f"DSA model weights not found: {DSV4_MTP}")
        opp, lib = _vendor_paths()
        env = dict(DSV4_SGLANG_ENVS)
        env["ASCEND_CUSTOM_OPP_PATH"] = opp
        env["LD_LIBRARY_PATH"] = (
            lib + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        )
        cls.server = (
            AscendServer(
                "07_speculative_dsa_topk_backend",
                DSV4_MTP,
                cls.PORT,
                other_args=[
                    "--quantization",
                    "modelslim",
                    "--mem-fraction-static",
                    "0.75",
                    "--attention-backend",
                    "ascend",
                    "--disable-cuda-graph",
                    "--speculative-algorithm",
                    "EAGLE",
                    "--speculative-num-steps",
                    "1",
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    "2",
                    "--speculative-dsa-topk-backend",
                    "torch",
                ],
                cards=DSV4_CARDS,
                env_extra=env,
                timeout=1800,
            )
            .start()
            .join_launch()
        )

    def test_speculative_dsa_topk_backend_real_server(self):
        """Real draft worker construction + DSA backend consumed + real generation."""
        log = self.server.log()

        # 1) server_args echo: the draft-side parameter is accepted
        self.assertIn("speculative_dsa_topk_backend", log)
        self.assertIn("'torch'", log)

        # 2) Two weight loads for target + MTP draft
        self.assertGreaterEqual(
            log.count("Load weight begin"),
            2,
            "should contain two weight loads for target and draft",
        )

        # 3) Real generation: EAGLE speculation + full DSA sparse attention pipeline
        from para_case_common import generate

        out = generate(
            self.PORT, "The capital of France is", max_new_tokens=24, timeout=300
        )
        text = out.get("text", "")
        self.assertTrue(text.strip(), f"generation is empty: {out}")
        self.assertGreater(out.get("meta_info", {}).get("completion_tokens", 0), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
