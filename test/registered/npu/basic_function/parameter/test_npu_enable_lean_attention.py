"""--enable-lean-attention: Lean (Work-Centric) Attention decode kernel.

Verification level: no_effect (pure no-op on NPU)
Consumption points: layers/attention/triton_backend.py:180, :2231 (the only consumption points in the repo)
Server launch style: sglang/test/ascend (popen_launch_server + kill_process_tree).
Evidence: the only consumption point in the repo is TritonAttnBackend; locally
--attention-backend ascend instantiates AscendAttnBackend / AscendHybridLinearAttnBackend,
neither of which reads this parameter.
After server launch, /server_info shows enable_lean_attention=true and attention_backend=ascend,
and besides the command-line echo the log has no lean traces at all → the parameter
is parsed successfully but has zero effect.

[Test Category] Parameter
[Test Target] --enable-lean-attention
"""

import unittest

from para_case_common import (
    COMMON_OTHER_ARGS,
    SMALL,
    AscendServer,
    AscendServerTestCase,
    generate,
    server_info,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=900, suite="nightly-1-npu-a3", nightly=True)


class TestEnableLeanAttention(AscendServerTestCase):
    PORT = 31190

    @classmethod
    def setUpClass(cls):
        cls.server = AscendServer(
            "05_enable_lean_attention",
            SMALL,
            cls.PORT,
            other_args=COMMON_OTHER_ARGS + ["--enable-lean-attention"],
            timeout=900,
        ).start()
        cls.server.join_launch()

    def test_enable_lean_attention_no_effect_on_npu(self):
        """The parameter is resolved but changes no behavior on NPU (TritonAttnBackend is never constructed)."""
        info = server_info(self.PORT)
        self.assertTrue(info["enable_lean_attention"])
        self.assertEqual(info["attention_backend"], "ascend")
        out = generate(self.PORT, "The capital of France is")
        self.assertTrue(out["text"])

        # No lean attention runtime logs should appear in the server launch log
        # (excluding the launch command and config echo of command=/server_args=)
        log = self.server.log()
        lean_lines = [
            l
            for l in log.splitlines()
            if "lean" in l.lower()
            and "server_args=" not in l
            and not l.startswith("command=")
        ]
        self.assertFalse(
            lean_lines,
            f"No lean attention runtime logs expected on NPU: {lean_lines[:3]}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
