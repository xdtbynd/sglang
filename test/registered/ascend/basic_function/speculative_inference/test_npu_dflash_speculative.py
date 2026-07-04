"""Test DFLASH speculative decoding on NPU.

[Test Category] Speculative Decoding
[Test Target] --speculative-algorithm=DFLASH;
--speculative-draft-model-path;
--speculative-dflash-block-size;
--speculative-draft-window-size;
--speculative-draft-attention-backend
[Platform] NPU (Ascend A3, CANN 9.0.0)
[Porting Source] New test case (no GPU counterpart for DFLASH on NPU CI)
"""

import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_8B_DFLASH_B16_WEIGHTS_PATH,
    QWEN3_8B_WEIGHTS_PATH,
    logger,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


NPU_ENV = {
    **os.environ,
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "ASCEND_USE_FIA": "0",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
}


class TestNPUDFlashSpeculative(CustomTestCase):
    """Test DFLASH speculative decoding on NPU.

    DFLASH (Draft Flash) is a speculative decoding algorithm that uses a
    block-structured draft model. This test verifies:
    1. Server starts successfully with DFLASH parameters
    2. Basic inference produces correct output
    3. GSM8K accuracy meets threshold
    4. Speculative accept length is reasonable (> 1.0)
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_8B_WEIGHTS_PATH
        cls.draft_model = QWEN3_8B_DFLASH_B16_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

        launch_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--disable-radix-cache",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.8",
            "--tp-size",
            "1",
            "--dp-size",
            "1",
            "--sampling-backend",
            "ascend",
            "--max-running-requests",
            "32",
            "--chunked-prefill-size",
            "-1",
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-draft-model-path",
            cls.draft_model,
            "--speculative-dflash-block-size",
            "16",
            "--speculative-draft-attention-backend",
            "ascend",
        ]

        logger.info("Starting DFLASH speculative server on NPU...")
        logger.info("Model: %s", cls.model)
        logger.info("Draft model: %s", cls.draft_model)
        logger.info("Launch args: %s", launch_args)

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=launch_args,
            env=NPU_ENV,
        )
        logger.info("DFLASH server started successfully.")

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_server_info(self):
        """Verify DFLASH configuration is reflected in /server_info."""
        resp = requests.get(self.base_url + "/server_info", timeout=30)
        self.assertEqual(resp.status_code, 200)
        info = resp.json()

        logger.info(
            "Server info: %s",
            {
                k: info.get(k)
                for k in [
                    "speculative_algorithm",
                    "speculative_dflash_block_size",
                    "speculative_num_steps",
                    "speculative_num_draft_tokens",
                ]
            },
        )

        self.assertEqual(
            info.get("speculative_algorithm"),
            "DFLASH",
            "speculative_algorithm should be DFLASH",
        )
        self.assertEqual(
            info.get("speculative_dflash_block_size"),
            16,
            "speculative_dflash_block_size should be 16",
        )

    def test_b_basic_inference(self):
        """Verify basic inference works with DFLASH."""
        from urllib.parse import urlparse

        parsed = urlparse(self.base_url)
        args = BenchArgs(
            host=parsed.hostname,
            port=parsed.port,
        )
        response = send_one_prompt(args, print_output=False)
        self.assertIsNotNone(response)
        self.assertGreater(len(response), 0)
        logger.info("Basic inference response: %s", response[:200])

    def test_c_gsm8k(self):
        """Verify GSM8K accuracy with DFLASH speculative decoding."""
        requests.get(self.base_url + "/flush_cache", timeout=30)

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)

        logger.info("GSM8K metrics: %s", metrics)

        server_info = requests.get(self.base_url + "/server_info", timeout=30).json()
        avg_spec_accept_length = None
        if "internal_states" in server_info and len(server_info["internal_states"]) > 0:
            internal_state = server_info["internal_states"][0]
            if "avg_spec_accept_length" in internal_state:
                avg_spec_accept_length = internal_state["avg_spec_accept_length"]
            elif "spec_accept_length" in internal_state:
                avg_spec_accept_length = internal_state["spec_accept_length"]

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (DFLASH on NPU)\n"
                f'{metrics["score"]=:.3f}\n'
                f"{avg_spec_accept_length=}\n"
            )

        self.assertGreater(metrics["score"], 0.55, "GSM8K score should be > 0.55")
        if avg_spec_accept_length is not None:
            self.assertGreater(
                avg_spec_accept_length,
                1.0,
                "avg_spec_accept_length should be > 1.0 for DFLASH to be beneficial",
            )


if __name__ == "__main__":
    unittest.main()
