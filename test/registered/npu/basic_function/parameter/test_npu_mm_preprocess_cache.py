"""--mm-preprocess-cache-size-mb / --trust-mm-content-hashes: multimodal preprocess cache parameters.

[Test Category] Parameter
[Test Target] --mm-preprocess-cache-size-mb, --trust-mm-content-hashes

--mm-preprocess-cache-size-mb (verification level: confirmed)
Consumption points: multimodal/processors/base_processor.py:275,
arg_groups/model_overrides/qwen3_vl.py:101; validated in arg_groups/serving_hook.py:121.
Evidence: the budget really determines the cache construction and capacity; log
`Multimodal preprocess cache enabled for ...: 256 MiB total (256 MiB per
tokenizer worker), at most 8192 entries`.

--trust-mm-content-hashes (verification level: confirmed)
Consumption points: multimodal/processors/base_processor.py:287,
multimodal/media_artifacts/base.py:311.
Evidence: the switch really toggles the hash trust mode of the cache; the
corresponding field in the log changes from `verified` to `trusted`
(`caller content hashes are trusted.`).

Both test cases share the HYBRID model + --page-size 1 --disable-overlap-schedule
base; trust-mm-content-hashes is verified on top of mm-preprocess-cache-size-mb.
Server launch style: sglang/test/ascend (popen_launch_server + kill_process_tree).
"""

import unittest

from para_case_common import (
    COMMON_OTHER_ARGS,
    HYBRID,
    AscendServer,
    AscendServerTestCase,
    server_info,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=3600, suite="nightly-1-npu-a3", nightly=True)


class TestMmPreprocessCacheSizeMb(AscendServerTestCase):
    PORT = 31330

    @classmethod
    def setUpClass(cls):
        cls.server = AscendServer(
            "23_mm_preprocess_cache_size_mb",
            HYBRID,
            cls.PORT,
            other_args=COMMON_OTHER_ARGS
            + [
                "--mm-preprocess-cache-size-mb",
                "256",
                "--page-size",
                "1",
                "--disable-overlap-schedule",
            ],
            timeout=1800,
        ).start()
        cls.server.join_launch()

    def test_mm_preprocess_cache_budget(self):
        """--mm-preprocess-cache-size-mb=256: the cache is enabled per the budget and split evenly across workers."""
        info = server_info(self.PORT)
        self.assertEqual(info["mm_preprocess_cache_size_mb"], 256)
        log = self.server.log()
        marker = "Multimodal preprocess cache enabled for"
        self.assertIn(marker, log)
        line = [l for l in log.splitlines() if marker in l][-1]
        self.assertIn("256 MiB total", line)


class TestTrustMmContentHashes(AscendServerTestCase):
    PORT = 31340

    @classmethod
    def setUpClass(cls):
        cls.server = AscendServer(
            "24_trust_mm_content_hashes",
            HYBRID,
            cls.PORT,
            other_args=COMMON_OTHER_ARGS
            + [
                "--mm-preprocess-cache-size-mb",
                "256",
                "--trust-mm-content-hashes",
                "--page-size",
                "1",
                "--disable-overlap-schedule",
            ],
            timeout=1800,
        ).start()
        cls.server.join_launch()

    def test_trust_mm_content_hashes_trusted_mode(self):
        """--trust-mm-content-hashes: the cache trust mode switches to trusted."""
        info = server_info(self.PORT)
        self.assertTrue(info["trust_mm_content_hashes"])
        log = self.server.log()
        line = [l for l in log.splitlines() if "caller content hashes are" in l][-1]
        self.assertIn("trusted", line)


if __name__ == "__main__":
    unittest.main(verbosity=2)
