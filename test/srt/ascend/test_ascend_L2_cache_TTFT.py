import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    run_bench_serving,
)

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestL2Cache(CustomTestCase):

    def test_L2_cache_TTFT(self):
        """After enabling L1L2 cache,the TTFT improved by 40% compared to when it was not enabled"""
        TTFTS = []
        model = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-32B"
        base_url = DEFAULT_URL_FOR_TEST
        other_args_list = [
            [
                "--trust-remote-code",
                "--max-running-requests",
                16,
                "--disable-radix-cache",
                "--chunked-prefill-size",
                "-1",
                "--mem-fraction-static",
                0.8,
                "--tp-size",
                4,
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ],
            [
                "--trust-remote-code",
                "--max-running-requests",
                16,
                "--chunked-prefill-size",
                "-1",
                "--mem-fraction-static",
                0.8,
                "--tp-size",
                4,
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--enable-hierarchical-cache",
                "--hicache-ratio",
                5,
                "--hicache-write-policy",
                "write_through",
            ]
        ]
        for other_args in other_args_list:
            res = run_bench_serving(
                model=model,
                dataset_name="generated-shared-prefix",
                num_prompts=128,
                random_input_len=3584,
                random_output_len=1,
                requests_rate=float("inf"),
                max_concurrency=16,
                gsp_num_groups=1,
                gsp_prompts_per_group=128,
                gsp_system_prompt_len=1792,
                gsp_question_len=1792,
                gsp_output_len=1,
                other_server_args=other_args,
            )
            TTFT = res["mean_ttft_ms"]
            TTFTS.append(TTFT)
        print("close and open L1L2 TTFT is " ,TTFTS)
        assert float(TTFTS[1]) <= 0.6*float(TTFTS[0])


if __name__ == "__main__":
    unittest.main()
