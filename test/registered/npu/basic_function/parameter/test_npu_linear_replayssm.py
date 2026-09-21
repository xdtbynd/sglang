"""--enable-linear-replayssm / --linear-replayssm-cache-len / --enable-linear-replayssm-spec: ReplaySSM family parameters.

[Test Category] Parameter
[Test Target] --enable-linear-replayssm, --linear-replayssm-cache-len,
              --enable-linear-replayssm-spec

--enable-linear-replayssm (verification level: no_effect, pure memory overhead on NPU)
Code locations: arg_groups/fields/exec_.py:428 (definition),
layers/attention/linear/ascend_gdn_backend.py:108-162 (expected consumption points),
gdn_triton.py:44 (supports_packed_decode=False gate).
Evidence: ring buffers are genuinely allocated (about 6.8GB total for d/k/g on Qwen3.5-4B), but on NPU the
AscendGDNAttnBackend.forward_decode never reads the ring parameters, and
supports_packed_decode=False blocks the packed_decode branch → zero speedup, memory consumed for nothing.

--linear-replayssm-cache-len (verification level: partial, the value genuinely determines the allocation,
but the ReplaySSM decode it serves has no effect on NPU)
Consumption points: mem_cache/memory_pool.py (uses record_len when allocating the ring).
Evidence: the value directly determines the ring's record_len and allocation size; the d component scales strictly linearly
(historical three-way comparison: 8→1.148GB / 32→4.559GB / 64→9.188GB, diagnostic logs in
local/para_cases/logs/diag_12_len_*.log). This case uses L=32 for single-point verification.

--enable-linear-replayssm-spec (verification level: partial, kernel-limited, root cause is
a triton-ascend toolchain bug)
Consumption points: arg_groups/attention_hook.py:361 (derives --mamba-ssm-dtype float32),
layers/attention/linear/kda_backend.py:495, disaggregation/decode.py:255.
Effective part: genuinely derives mamba_ssm_dtype=float32 and allocates the spec-specific window
(rawv/rawk per-slot raw input windows).
Ineffective part: real spec verify cannot run on Ascend — the failing kernel is
_advance_gdn_spec_cursors_kernel (gdn_replayssm_spec_decode.py); a minimized reproduction
(local/para_cases/logs/repro_gdn_nrows.log) proves that only n_rows=1 triggers the
triton-ascend assertion `addptrRes.hasOneUse()` failure while n_rows>=2 works normally;
--page-size 1 avoids that kernel but hits the npu_fused_infer_attention_score
(head_dim=256,TND) limitation.

Server launch style: sglang/test/ascend (popen_launch_server + kill_process_tree).
"""

import unittest

from para_case_common import (
    COMMON_OTHER_ARGS,
    HYBRID,
    AscendServer,
    AscendServerTestCase,
    generate,
    server_info,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=5400, suite="nightly-1-npu-a3", nightly=True)


class TestEnableLinearReplayssm(AscendServerTestCase):
    PORT = 31240

    @classmethod
    def setUpClass(cls):
        cls.server = AscendServer(
            "11_enable_linear_replayssm",
            HYBRID,
            cls.PORT,
            other_args=COMMON_OTHER_ARGS
            + [
                "--enable-linear-replayssm",
                "--page-size",
                "1",
                "--disable-overlap-schedule",
            ],
            timeout=1800,
        ).start()
        cls.server.join_launch()

    def test_enable_linear_replayssm_alloc_only(self):
        """--enable-linear-replayssm: ring buffers are genuinely allocated, but the decode kernel does not consume them."""
        info = server_info(self.PORT)
        self.assertTrue(info["enable_linear_replayssm"])
        log = self.server.log()
        marker = "GDN ReplaySSM ring buffers allocated (record_len="
        self.assertIn(marker, log)
        line = [l for l in log.splitlines() if marker in l][-1]
        self.assertIn("fold=False", line)
        out = generate(self.PORT, "ReplaySSM ring test: count to five ->")
        self.assertTrue(out["text"])


class TestLinearReplayssmCacheLen(AscendServerTestCase):
    PORT = 31250
    CACHE_LEN = 32

    @classmethod
    def setUpClass(cls):
        cls.server = AscendServer(
            "12_linear_replayssm_cache_len",
            HYBRID,
            cls.PORT,
            other_args=COMMON_OTHER_ARGS
            + [
                "--enable-linear-replayssm",
                "--linear-replayssm-cache-len",
                str(cls.CACHE_LEN),
                "--page-size",
                "1",
                "--disable-overlap-schedule",
            ],
            timeout=1800,
        ).start()
        cls.server.join_launch()

    def test_cache_len_sets_record_len(self):
        """--linear-replayssm-cache-len=32: directly determines the ring's record_len."""
        info = server_info(self.PORT)
        self.assertEqual(info["linear_replayssm_cache_len"], self.CACHE_LEN)
        log = self.server.log()
        marker = f"GDN ReplaySSM ring buffers allocated (record_len={self.CACHE_LEN},"
        self.assertIn(marker, log)


class TestEnableLinearReplayssmSpec(AscendServerTestCase):
    PORT = 31260

    @classmethod
    def setUpClass(cls):
        # Without speculative decoding, max_running_requests is not derived yet, and
        # the replayssm branch in kv_cache_configurator.py:2449 reads it directly
        # (TypeError: NoneType // int), so we set it explicitly to complete the server launch verification.
        cls.server = AscendServer(
            "13_replayssm_spec",
            HYBRID,
            cls.PORT,
            other_args=COMMON_OTHER_ARGS
            + [
                "--linear-attn-backend",
                "triton",
                "--enable-linear-replayssm-spec",
                "--linear-replayssm-cache-len",
                "16",
                "--speculative-dsa-topk-backend",
                "torch",
                "--max-running-requests",
                "8",
            ],
            env_extra={"SGLANG_RAGGED_VERIFY_MODE": "static"},
            timeout=2400,
        ).start()
        cls.server.join_launch()

    def test_spec_derivation_and_spec_ring(self):
        """Automatic fp32 + spec-specific ring buffers (rawv/rawk) are genuinely assembled."""
        info = server_info(self.PORT)
        self.assertTrue(info["enable_linear_replayssm_spec"])
        self.assertFalse(info["enable_linear_replayssm"])
        self.assertEqual(info["mamba_ssm_dtype"], "float32")
        self.assertEqual(info["linear_replayssm_cache_len"], 16)

        log = self.server.log()
        self.assertIn(
            "--enable-linear-replayssm-spec: setting --mamba-ssm-dtype float32", log
        )
        marker = "GDN ReplaySSM ring buffers allocated (record_len=16, fold=False)"
        line = [l for l in log.splitlines() if marker in l][-1]
        # rawv/rawk are the per-slot raw input windows specific to spec-verify
        self.assertIn("rawv=", line)
        self.assertIn("rawk=", line)

        out = generate(self.PORT, "ReplaySSM spec ring ->", max_new_tokens=16)
        self.assertTrue(out["text"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
