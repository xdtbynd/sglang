"""--linear-attn-backend: the default kernel backend for linear attention (GDN/KDA).

Verification level: confirmed
Consumption points: layers/attention/attention_registry.py:428 (resolve_linear_attn_backends)
→ AscendGDNAttnBackend.__init__ constructs GDNKernelDispatcher
Server launch style: sglang/test/ascend (popen_launch_server + kill_process_tree).
Evidence: the log line `GDN kernel dispatcher: decode=... extend=... verify=...` proves
that the value genuinely determines which kernel classes the dispatcher selects.
Note: on NPU, TritonGDNKernel's chunk/update is actually replaced by the sgl_kernel_npu
implementation (gdn_triton.py:20-27); the name is triton but the kernels come from NPU.

[Test Category] Parameter
[Test Target] --linear-attn-backend
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

register_npu_ci(est_time=1800, suite="nightly-1-npu-a3", nightly=True)


class TestLinearAttnBackend(AscendServerTestCase):
    PORT = 31230

    @classmethod
    def setUpClass(cls):
        cls.server = AscendServer(
            "10_linear_attn_backend",
            HYBRID,
            cls.PORT,
            other_args=COMMON_OTHER_ARGS
            + [
                "--linear-attn-backend",
                "triton",
                "--page-size",
                "1",
                "--disable-overlap-schedule",
            ],
            timeout=1800,
        ).start()
        cls.server.join_launch()

    def test_linear_attn_backend_triton(self):
        """--linear-attn-backend=triton: decode/prefill backends are genuinely dispatched."""
        info = server_info(self.PORT)
        self.assertEqual(info["linear_attn_backend"], "triton")
        log = self.server.log()
        self.assertIn("GDN kernel dispatcher: decode=", log)
        line = [l for l in log.splitlines() if "GDN kernel dispatcher: decode=" in l][
            -1
        ]
        self.assertIn("triton", line.lower())
        out = generate(self.PORT, "linear attn backend probe ->")
        self.assertTrue(out["text"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
