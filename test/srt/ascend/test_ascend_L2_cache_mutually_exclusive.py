import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="stage-b-test-2-npu-a2", nightly=False)
register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class TestL2Cache(CustomTestCase):
    def test_L2_cache_mutually_exclusive(cls):
        """The arguments enable-hierarchical-cache and disable-radix-cache are mutually exclusive,service startup failure"""
        error_message = "The arguments enable-hierarchical-cache and disable-radix-cache are mutually exclusive"
        model = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-32B"
        base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--enable-hierarchical-cache",
            "--disable-radix-cache",
            "--mem-fraction-static",
            0.8,
            "--tp-size",
            2,
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]
        out_log_file = open("./cache_out_log.txt", "w+", encoding="utf-8")
        err_log_file = open("./cache_err_log.txt", "w+", encoding="utf-8")
        try:
            process = popen_launch_server(
                model,
                base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=other_args,
                return_stdout=(out_log_file, err_log_file),
            )
        except Exception as e:
            print(f"Server launch failed: {e}")
        finally:
            err_log_file.seek(0)
            content = err_log_file.read()
            self.assertIn(error_message, content)
            err_log_file.close()
            out_log_file.close()
            os.remove("./cache_out_log.txt")
            os.remove("./cache_err_log.txt")


if __name__ == "__main__":
    unittest.main()
