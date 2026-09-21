"""--load-publish-endpoint: runtime load PUB socket for load-aware routing to subscribe to.

Verification level: confirmed
Consumption points: disaggregation/kv_events.py:171,
managers/scheduler_components/load_publisher.py:136, managers/scheduler.py:2409
Validation: arg_groups/validation_hook.py:323 (must be used together with --kv-events-config)
Server launch style: sglang/test/ascend (popen_launch_server + kill_process_tree).

[Test Category] Parameter
[Test Target] --load-publish-endpoint
"""

import json
import unittest

from para_case_common import (
    COMMON_OTHER_ARGS,
    SMALL,
    AscendServer,
    AscendServerTestCase,
    port_listening,
    server_info,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=900, suite="nightly-1-npu-a3", nightly=True)


class TestLoadPublishEndpoint(AscendServerTestCase):
    PORT = 31170
    KV_PORT = 5592

    @classmethod
    def setUpClass(cls):
        cls.server = AscendServer(
            "03_load_publish_endpoint",
            SMALL,
            cls.PORT,
            other_args=COMMON_OTHER_ARGS
            + [
                "--kv-events-config",
                json.dumps(
                    {
                        "publisher": "zmq",
                        "endpoint": f"tcp://*:{cls.KV_PORT}",
                        "topic": "kv",
                    }
                ),
                "--load-publish-endpoint",
                "auto",
            ],
            timeout=900,
        ).start()
        cls.server.join_launch()

    def test_load_publish_endpoint_auto(self):
        """--load-publish-endpoint=auto: load PUB socket bound and advertised via /server_info."""
        info = server_info(self.PORT)
        kv = info.get("kv_events")
        self.assertIsNotNone(
            kv, "kv_events descriptor missing (kv-events-config did not take effect)"
        )
        base = kv.get("load_endpoint_port_base")
        self.assertIsNotNone(base, f"load_endpoint_port_base missing: {kv}")
        self.assertEqual(kv.get("load_topic"), "load")
        # auto should place base after the kv port range: kv_base + dp_size
        self.assertEqual(base, self.KV_PORT + kv["dp_size"])
        for rank in range(kv["dp_size"]):
            self.assertTrue(
                port_listening(base + rank),
                f"load publisher rank {rank} port {base + rank} not listening",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
