"""--disaggregation-decode-retraction-backup: KV backup backend used when a
PD decode server retracts requests.

[Test Category] Parameter
[Test Target] --disaggregation-decode-retraction-backup

Launches a real server with `--disaggregation-decode-retraction-backup
cpu_tensor` and verifies that the value lands in the runtime config and the
server keeps serving.

Consumption points: mem_cache/kv_cache_builder.py:148,
managers/scheduler.py:5378, disaggregation/decode.py:827,
managers/schedule_batch.py:2188.

Note: the alternative value `host_pool` requires a PD decode topology with a
HiCache host pool, which a single-node launch cannot provide — that rejection
is kept as a parser-level check below. The backup/restore/discard dispatch
(offload -> load via req.offload_kv_cache / load_kv_cache) runs inside the
real retraction path on a PD decode server.
"""

import argparse
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


class TestDisaggregationDecodeRetractionBackup(AscendServerTestCase):
    PORT = 31400

    @classmethod
    def setUpClass(cls):
        cls.server = AscendServer(
            "19_disaggregation_decode_retraction_backup",
            SMALL,
            cls.PORT,
            other_args=COMMON_OTHER_ARGS
            + ["--disaggregation-decode-retraction-backup", "cpu_tensor"],
            timeout=900,
        ).start()
        cls.server.join_launch()

    def test_retraction_backup_backend_in_runtime_config(self):
        """cpu_tensor lands in the runtime config and the server keeps serving."""
        info = server_info(self.PORT)
        self.assertEqual(info["disaggregation_decode_retraction_backup"], "cpu_tensor")

        out = generate(self.PORT, "Retraction backup smoke: count to five ->")
        self.assertTrue(out["text"])

    def test_host_pool_rejected_without_pd_decode(self):
        """host_pool is only allowed on a PD decode server (parser-level)."""
        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser(add_help=False)
        ServerArgs.add_cli_args(parser)
        with self.assertRaises(ValueError) as ctx:
            ServerArgs.from_cli_args(
                parser.parse_args(
                    [
                        "--model-path",
                        SMALL,
                        "--disaggregation-decode-retraction-backup",
                        "host_pool",
                    ]
                )
            )
        self.assertIn("only supported on a PD decode server", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
