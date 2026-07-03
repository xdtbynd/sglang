"""Test Adaptive Speculative Decoding on NPU.

[Test Category] Speculative Decoding
[Test Target] --speculative-algorithm=EAGLE3; --speculative-adaptive;
--speculative-adaptive-config; --speculative-num-steps (dynamic);
--speculative-eagle-topk; --speculative-num-draft-tokens
[Platform] NPU (Ascend A3, CANN 9.0.0)
[Porting Source] Ported from GPU test: sgl-project/sglang test/test_adaptive_speculative.py
  Class: TestAdaptiveSpeculativeServer

Porting notes:
  - attention-backend: triton -> ascend
  - model: Qwen2.5-1.5B-Instruct -> Qwen3-8B (NPU CI pre-installed)
  - mem-fraction-static: 0.85 -> 0.7 (NPU standard)
  - GSM8K threshold: 0.20 -> 0.69 (stricter, consistent with NPU spec tests)
  - Added NPU env vars (SGLANG_ENABLE_SPEC_V2, etc.)
  - register_cuda_ci -> register_npu_ci
  - print() -> logger.info()
  - Added --disable-cuda-graph (NPU doesn't support CUDA Graph)
  - Added --sampling-backend ascend
  - TestAdaptiveZeroStepBatchSizeServer NOT ported (depends on GPU routing logic)
"""

import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_8B_EAGLE3_WEIGHTS_PATH,
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


class TestNPUAdaptiveSpeculativeServer(CustomTestCase):
    """Test Adaptive Speculative Decoding on NPU.

    Ported from GPU: sgl-project/sglang test/test_adaptive_speculative.py
    Class: TestAdaptiveSpeculativeServer

    This test verifies that the adaptive speculative decoding system can:
    1. Dynamically increase speculative_num_steps (upshift) when accept rate is high
    2. Dynamically decrease speculative_num_steps (downshift) when accept rate is low
    3. Maintain GSM8K accuracy after step switches
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_8B_WEIGHTS_PATH
        cls.draft_model = QWEN3_8B_EAGLE3_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

        launch_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.7",
            "--tp-size",
            "1",
            "--sampling-backend",
            "ascend",
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            cls.draft_model,
            "--speculative-num-steps",
            "1",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "8",
        ]

        logger.info("Starting Adaptive Speculative server on NPU...")
        logger.info("Model: %s", cls.model)
        logger.info("Draft model: %s", cls.draft_model)

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=launch_args,
            env=NPU_ENV,
        )
        logger.info("Adaptive server started successfully.")

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _get_server_info(self):
        resp = requests.get(self.base_url + "/server_info", timeout=30)
        self.assertEqual(resp.status_code, 200)
        return resp.json()

    def _get_speculative_num_steps(self):
        info = self._get_server_info()
        num_steps = info.get("speculative_num_steps")
        logger.info("Current speculative_num_steps: %s", num_steps)
        return num_steps

    def _drive_upshift(self):
        """Drive the system to upshift (increase speculative_num_steps).

        1. Set target speculative_num_steps to 3 via /set_args
        2. Send prompts to trigger actual inference (high accept rate)
        3. Verify speculative_num_steps has increased to 3
        """
        logger.info("=== Driving Upshift (1 -> 3) ===")

        # Set target num_steps to 3
        resp = requests.post(
            self.base_url + "/set_args",
            json={"speculative_num_steps": 3},
            timeout=30,
        )
        self.assertEqual(resp.status_code, 200)
        logger.info("Set speculative_num_steps=3 via /set_args")

        # Send prompts to trigger adaptive upshift
        args = BenchArgs(base_url=self.base_url, model=self.model)
        for i in range(5):
            send_one_prompt(args)
            logger.info("Sent prompt %d/5 for upshift", i + 1)

        # Verify upshift
        num_steps = self._get_speculative_num_steps()
        self.assertEqual(
            num_steps, 3,
            f"Expected speculative_num_steps=3 after upshift, got {num_steps}"
        )
        logger.info("Upshift verified: speculative_num_steps=3")
        return {"speculative_num_steps": num_steps}

    def _drive_downshift(self):
        """Drive the system to downshift (decrease speculative_num_steps).

        1. Set target speculative_num_steps to 1 via /set_args
        2. Send prompts to trigger actual inference (low accept rate)
        3. Verify speculative_num_steps has decreased to 1
        """
        logger.info("=== Driving Downshift (3 -> 1) ===")

        # Set target num_steps to 1
        resp = requests.post(
            self.base_url + "/set_args",
            json={"speculative_num_steps": 1},
            timeout=30,
        )
        self.assertEqual(resp.status_code, 200)
        logger.info("Set speculative_num_steps=1 via /set_args")

        # Send prompts to trigger adaptive downshift
        args = BenchArgs(base_url=self.base_url, model=self.model)
        for i in range(5):
            send_one_prompt(args)
            logger.info("Sent prompt %d/5 for downshift", i + 1)

        # Verify downshift
        num_steps = self._get_speculative_num_steps()
        self.assertEqual(
            num_steps, 1,
            f"Expected speculative_num_steps=1 after downshift, got {num_steps}"
        )
        logger.info("Downshift verified: speculative_num_steps=1")
        return {"speculative_num_steps": num_steps}

    def test_adaptive_switches_and_gsm8k(self):
        """Main test: upshift -> downshift -> gsm8k accuracy check.

        Ported from GPU: test_gsm8k_after_adaptive_switches
        """
        # Step 1: Drive upshift (1 -> 3)
        state = self._drive_upshift()
        self.assertEqual(state["speculative_num_steps"], 3)

        # Step 2: Drive downshift (3 -> 1)
        state = self._drive_downshift()
        self.assertEqual(state["speculative_num_steps"], 1)

        # Step 3: Run GSM8K eval to verify accuracy after switches
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
        logger.info("GSM8K metrics after adaptive switches: %s", metrics)

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (Adaptive Speculative on NPU)\n"
                f'{metrics["score"]=:.3f}\n'
            )

        # NPU uses stricter threshold (0.69) than GPU (0.20)
        self.assertGreater(
            metrics["score"], 0.69,
            "GSM8K score should be > 0.69 after adaptive switches"
        )


if __name__ == "__main__":
    unittest.main()
