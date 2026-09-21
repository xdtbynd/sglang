"""--return-hidden-states-mode: the maximum hidden-state return mode allowed by the server.

Verification level: confirmed (both positive and negative)
Consumption points: arg_groups/serving_hook.py:655 (handle_return_hidden_states_mode) →
model_executor/runner/base_runner.py:230, model_executor/forward_batch_info.py:230
Server launch style: sglang/test/ascend (popen_launch_server + kill_process_tree).
- Positive: on an instance with mode=full, /generate with return_hidden_states=true
  really returns hidden_states for max_new_tokens steps.
- Negative: on an instance without this mode configured, the same request is rejected
  with HTTP 400 `not configured to return hidden states`.

[Test Category] Parameter
[Test Target] --return-hidden-states-mode
"""

import unittest

import requests
from para_case_common import (
    COMMON_OTHER_ARGS,
    SMALL,
    AscendServer,
    AscendServerTestCase,
    generate,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1200, suite="nightly-1-npu-a3", nightly=True)


class TestReturnHiddenStatesMode(AscendServerTestCase):
    POS_PORT = 31311
    NEG_PORT = 31310

    @classmethod
    def setUpClass(cls):
        cls.server = AscendServer(
            "18_return_hidden_states_full",
            SMALL,
            cls.POS_PORT,
            other_args=COMMON_OTHER_ARGS
            + ["--enable-return-hidden-states", "--return-hidden-states-mode", "full"],
            timeout=900,
        ).start()
        cls.server.join_launch()

    def test_positive_full_mode_returns_hidden_states(self):
        """--return-hidden-states-mode=full: requests can obtain hidden_states."""
        out = generate(
            self.POS_PORT,
            "The capital of France is",
            max_new_tokens=8,
            extra={"return_hidden_states": True},
        )
        hs = out.get("meta_info", {}).get("hidden_states")
        self.assertIsNotNone(
            hs, f"no hidden_states in meta_info: {list(out['meta_info'])}"
        )
        self.assertEqual(
            len(hs),
            8,
            f"hidden_states length should equal max_new_tokens=8, got {len(hs)}",
        )

    def test_negative_absent_mode_rejected(self):
        """Without the mode enabled, a request carrying return_hidden_states must be rejected by the server."""
        server = AscendServer(
            "18_return_hidden_states_absent",
            SMALL,
            self.NEG_PORT,
            other_args=COMMON_OTHER_ARGS,
            timeout=900,
        ).start()
        try:
            server.join_launch()
            payload = {
                "text": "hi",
                "sampling_params": {"max_new_tokens": 2, "temperature": 0},
                "return_hidden_states": True,
            }
            r = requests.post(
                f"http://127.0.0.1:{self.NEG_PORT}/generate",
                json=payload,
                timeout=120,
            )
            self.assertNotEqual(
                r.status_code,
                200,
                f"succeeded even though the mode is not enabled: {r.text[:500]}",
            )
            self.assertIn(
                "not configured to return hidden states",
                r.text,
                f"error message mismatch: {r.text[:800]}",
            )
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main(verbosity=2)
