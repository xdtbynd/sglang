"""--gated-launch-port: suspend the memory-hungry launch phase and resume after an external release.

Verification level: confirmed
Consumption points: distributed/bootstrap.py:140
Server launch style: sglang/test/ascend (popen_launch_server + kill_process_tree).
Timing: popen_launch_server's /health_generate wait is blocked by the gate (running in
a background thread); use AscendServer's dedicated launch method start_gated(gate_port)
to wait for the gate port and the log marker to appear, the main flow asserts the main
service is unavailable and then activate_gate() releases it, followed by join_launch()
waiting for the service to be truly ready.

[Test Category] Parameter
[Test Target] --gated-launch-port
"""

import time
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

register_npu_ci(est_time=900, suite="nightly-1-npu-a3", nightly=True)


class TestGatedLaunch(AscendServerTestCase):
    PORT = 31180
    GATE_PORT = 31181

    @classmethod
    def setUpClass(cls):
        cls.server = AscendServer(
            "04_gated_launch_port",
            SMALL,
            cls.PORT,
            other_args=COMMON_OTHER_ARGS + ["--gated-launch-port", str(cls.GATE_PORT)],
            timeout=900,
        ).start_gated(cls.GATE_PORT)

    def test_gated_launch_port_blocks_until_activate(self):
        """Server launch is blocked at the gate and released after POST /gate/activate."""
        # start_gated has already waited for this marker; grab the log directly to assert the gate control server is up
        log = self.server.log()
        self.assertIn("Gated launch waiting for activation.", log)
        self.assertIn("Gated launch control server started on", log)
        r = requests.get(f"http://127.0.0.1:{self.GATE_PORT}/health", timeout=5)
        self.assertEqual(r.status_code, 200)

        def main_blocked():
            """The main service is unavailable while the gate is not released (/health does not return 200)."""
            try:
                return (
                    requests.get(
                        f"http://127.0.0.1:{self.PORT}/health", timeout=3
                    ).status_code
                    != 200
                )
            except requests.RequestException:
                return True

        # Main service unavailable while blocked (observed continuously for 15s)
        blocked_samples = []
        for _ in range(5):
            blocked_samples.append(main_blocked())
            time.sleep(3)
        self.assertTrue(
            all(blocked_samples),
            f"main service available before gate release: {blocked_samples}",
        )

        # Release (repeated POST is idempotent)
        for _ in range(2):
            r = self.server.activate_gate(self.GATE_PORT)
            self.assertEqual(r.status_code, 200)

        # After release, popen_launch_server's health wait completes in the background thread
        self.server.join_launch()
        log = self.server.log()
        self.assertIn("Gated launch activated. elapsed=", log)
        self.assertEqual(
            requests.get(
                f"http://127.0.0.1:{self.PORT}/health", timeout=10
            ).status_code,
            200,
        )
        out = generate(self.PORT, "hi")
        self.assertTrue(out["text"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
