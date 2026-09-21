"""--hicache-host-memory-mode / --hicache-storage-prefetch-retry-poll-interval: HiCache extended parameters.

[Test Category] Parameter
[Test Target] --hicache-host-memory-mode,
              --hicache-storage-prefetch-retry-poll-interval

--hicache-host-memory-mode (verification level: confirmed)
Consumption points: mem_cache/hybrid_cache/hybrid_pool_assembler.py:363,
managers/scheduler.py:3092, mem_cache/buffer_mode/pipeline.py:244;
validation at arg_groups/hicache_hook.py:180.
Evidence: with value buffer_only, BufferModePipeline is genuinely constructed (that log line is printed only in the buffer_only
branch); the file backend genuinely writes to disk, and a second request with the same prompt hits with cached_tokens>0.
Note: the disk-write directory of the file backend comes from the SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR environment variable;
file_path in --hicache-storage-backend-extra-config is not read by the file backend
(backend_factory.py:161-162 only passes storage_config).

--hicache-storage-prefetch-retry-poll-interval (verification level: config_only)
Consumption points: managers/scheduler.py:3139 (_process_storage_prefetch_retries).
Evidence: the value genuinely enters the runtime configuration and is passed to the consuming function.
Gap: the value is only used in the race where the L3 prefetch misses and the backup is still being committed (requires a real storage
backend under high load so that the first check happens earlier than commit completion); this case does not trigger a retry, so the
runtime configuration placement is taken as the criterion.

Server launch style: sglang/test/ascend (popen_launch_server + kill_process_tree).
"""

import os
import shutil
import time
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

register_npu_ci(est_time=1800, suite="nightly-1-npu-a3", nightly=True)


class TestHicacheHostMemoryMode(AscendServerTestCase):
    PORT = 31280
    STORAGE_DIR = "/tmp/para_cases_hicache_15"

    @classmethod
    def setUpClass(cls):
        shutil.rmtree(cls.STORAGE_DIR, ignore_errors=True)
        cls.server = AscendServer(
            "15_hicache_host_memory_mode",
            SMALL,
            cls.PORT,
            other_args=COMMON_OTHER_ARGS
            + [
                "--enable-hierarchical-cache",
                "--hicache-size",
                "512",
                "--hicache-write-policy",
                "write_through",
                "--hicache-storage-backend",
                "file",
                "--hicache-host-memory-mode",
                "buffer_only",
            ],
            env_extra={"SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.STORAGE_DIR},
            timeout=900,
        ).start()
        cls.server.join_launch()

    def test_hicache_host_memory_mode_buffer_only(self):
        """--hicache-host-memory-mode=buffer_only: the second-level cache + file backend are genuinely assembled and write to disk."""
        info = server_info(self.PORT)
        self.assertEqual(info["hicache_host_memory_mode"], "buffer_only")
        self.assertTrue(info["enable_hierarchical_cache"])
        log = self.server.log()
        # buffer_only-specific evidence: BufferModePipeline is constructed only when
        # host memory mode=buffer_only (buffer_mode/pipeline.py:217-219)
        self.assertIn("BufferModePipeline anchor_lock_cap_tokens=", log)
        self.assertIn("Creating storage backend 'file'", log)
        self.assertIn("Created HiCacheFile storage directory at", log)

        prompt = ("hi cache acceptance test sentence. " * 120).strip()
        generate(self.PORT, prompt, max_new_tokens=4)
        # L3 (file storage) did receive writes: under buffer_only the host memory is only a transit,
        # so the data must ultimately be written into storage
        deadline = time.time() + 120
        files = []
        while time.time() < deadline:
            files = (
                os.listdir(self.STORAGE_DIR) if os.path.isdir(self.STORAGE_DIR) else []
            )
            if files:
                break
            time.sleep(2)
        self.assertTrue(
            files,
            f"nothing written to disk in HiCacheFile storage directory {self.STORAGE_DIR}",
        )

        # Same prompt again: the data is fetched back from storage (the read-back path of buffer_only)
        second = generate(self.PORT, prompt, max_new_tokens=4)
        cached_2 = second["meta_info"]["cached_tokens"]
        self.assertGreater(
            cached_2, 0, "the second request with the same prompt did not hit L2/L3"
        )


class TestHicachePrefetchRetryPollInterval(AscendServerTestCase):
    PORT = 31290

    @classmethod
    def setUpClass(cls):
        cls.server = AscendServer(
            "16_hicache_prefetch_retry",
            SMALL,
            cls.PORT,
            other_args=COMMON_OTHER_ARGS
            + [
                "--enable-hierarchical-cache",
                "--hicache-size",
                "512",
                "--hicache-write-policy",
                "write_through",
                "--hicache-storage-prefetch-retry-poll-interval",
                "2",
            ],
            timeout=900,
        ).start()
        cls.server.join_launch()

    def test_prefetch_retry_poll_interval_in_runtime_config(self):
        """--hicache-storage-prefetch-retry-poll-interval=2: the value enters the runtime configuration."""
        info = server_info(self.PORT)
        self.assertEqual(info["hicache_storage_prefetch_retry_poll_interval"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
