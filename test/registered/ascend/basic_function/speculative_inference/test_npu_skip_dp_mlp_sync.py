"""Test --speculative-skip-dp-mlp-sync on NPU (positive + negative).

[Test Category] Speculative Decoding
[Test Target] --speculative-skip-dp-mlp-sync;
--speculative-algorithm=EAGLE (positive) / DFLASH (negative);
--enable-dp-attention; --enable-dp-lm-head; --tp-size; --dp-size
[Platform] NPU (Ascend A3, CANN 9.0.0)
[Porting Source] New test case

Test strategy:
  - Positive: EAGLE + skip-dp-mlp-sync -> server starts, GSM8K passes
  - Negative: DFLASH + skip-dp-mlp-sync -> server crashes with Assert error,
    captured via subprocess.run, stderr asserted to contain expected message.
    This ensures CI does NOT fail when the assert is correctly triggered.
"""

import os
import subprocess
import sys
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_8B_DFLASH_B16_WEIGHTS_PATH,
    QWEN3_8B_EAGLE3_WEIGHTS_PATH,
    QWEN3_8B_WEIGHTS_PATH,
    logger,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_npu_ci(est_time=600, suite="full-4-npu-a3", nightly=True)


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
    "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24666",
    "HCCL_BUFFSIZE": "200",
    "HCCL_EXEC_TIMEOUT": "200",
}


class TestNPUSkipDPMLPSyncPositive(CustomTestCase):
    """Positive test: EAGLE + skip-dp-mlp-sync should work correctly.

    --speculative-skip-dp-mlp-sync skips the DP MLP synchronization step
    in EAGLE speculative decoding with DP attention. This improves
    performance by avoiding unnecessary cross-DP communication.

    Note: The assert in speculative_hook.py requires
    speculative_algorithm == "EAGLE" (not EAGLE3). We use the EAGLE3 draft
    model weights with --speculative-algorithm EAGLE, which the framework
    accepts because EAGLE3 is a superset of EAGLE.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_8B_WEIGHTS_PATH
        cls.draft_model = QWEN3_8B_EAGLE3_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

        launch_args = [
            "--trust-remote-code",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "2",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "3",
            "--speculative-draft-model-path",
            cls.draft_model,
            "--tp-size",
            "2",
            "--dp-size",
            "2",
            "--enable-dp-attention",
            "--enable-dp-lm-head",
            "--moe-dense-tp-size",
            "1",
            "--attention-backend",
            "ascend",
            "--sampling-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.7",
            "--speculative-skip-dp-mlp-sync",
        ]

        logger.info("Starting EAGLE + skip-dp-mlp-sync server on NPU...")
        logger.info("Model: %s", cls.model)
        logger.info("Draft model: %s", cls.draft_model)
        logger.info("TP=2, DP=2, skip_dp_mlp_sync=True")

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=launch_args,
            env=NPU_ENV,
        )
        logger.info("Server started successfully with skip-dp-mlp-sync.")

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        """Verify GSM8K accuracy with skip-dp-mlp-sync enabled."""
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
        logger.info("GSM8K metrics (skip-dp-mlp-sync): %s", metrics)

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
                f"### test_gsm8k (EAGLE skip-dp-mlp-sync on NPU)\n"
                f'{metrics["score"]=:.3f}\n'
                f"{avg_spec_accept_length=}\n"
            )

        self.assertGreater(
            metrics["score"], 0.69, "GSM8K score should be > 0.69 with skip-dp-mlp-sync"
        )


class TestNPUSkipDPMLPSyncNegative(unittest.TestCase):
    """Negative test: non-EAGLE algorithm + skip-dp-mlp-sync should be rejected.

    --speculative-skip-dp-mlp-sync is only supported with EAGLE algorithm
    (not EAGLE3, not DFLASH). When used with DFLASH, the server should
    assert and refuse to start.

    This test uses subprocess.run to capture the crash output and verify
    the expected assert message, ensuring CI does NOT fail.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_8B_DFLASH_B16_WEIGHTS_PATH
        cls.base_url = "http://127.0.0.1:39999"

    def test_non_eagle_algorithm_rejected(self):
        """DFLASH + skip-dp-mlp-sync should trigger Assert and crash."""
        logger.info("=== Negative test: DFLASH + skip-dp-mlp-sync ===")
        logger.info("Expecting server to crash with Assert error.")

        env = {**NPU_ENV, "ASCEND_RT_VISIBLE_DEVICES": "0,1"}

        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.model,
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.7",
            "--tp-size",
            "2",
            "--dp-size",
            "2",
            "--enable-dp-attention",
            "--enable-dp-lm-head",
            "--moe-dense-tp-size",
            "1",
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-draft-model-path",
            self.model,
            "--speculative-dflash-block-size",
            "16",
            "--speculative-skip-dp-mlp-sync",
            "--host",
            "127.0.0.1",
            "--port",
            "39999",
        ]

        logger.info("Command: %s", " ".join(cmd))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )

        combined_output = result.stderr + result.stdout
        logger.info("Return code: %d", result.returncode)
        logger.info("Stderr (last 1000 chars): %s", result.stderr[-1000:])
        logger.info("Stdout (last 1000 chars): %s", result.stdout[-1000:])

        # Server should have crashed (non-zero exit)
        self.assertNotEqual(
            result.returncode,
            0,
            "Server should have crashed with Assert error for non-EAGLE + skip-dp-mlp-sync",
        )

        # Verify the expected Assert message is present
        # The assert message mentions skip_dp_mlp_sync is only for EAGLE
        possible_messages = [
            "skip_dp_mlp_sync",
            "speculative_algorithm == EAGLE",
            "only supported",
        ]

        found = False
        for msg in possible_messages:
            if msg.lower() in combined_output.lower():
                found = True
                logger.info("Found expected message: '%s'", msg)
                break

        self.assertTrue(
            found,
            f"Expected Assert message about skip_dp_mlp_sync not found in output. "
            f"Possible messages: {possible_messages}. "
            f"Output (last 2000 chars): {combined_output[-2000:]}",
        )

        logger.info("Negative test passed: Assert correctly intercepted.")


if __name__ == "__main__":
    unittest.main()
