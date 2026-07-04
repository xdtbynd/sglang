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
  - model: DEFAULT_TARGET_MODEL_EAGLE -> Qwen3-8B (NPU CI pre-installed)
  - draft model: DEFAULT_DRAFT_MODEL_EAGLE -> Qwen3-8B-EAGLE3
  - algorithm: EAGLE -> EAGLE3 (NPU preferred)
  - mem-fraction-static: 0.7 (unchanged)
  - GSM8K threshold: 0.20 -> 0.69 (stricter, consistent with NPU spec tests)
  - GSM8K num_examples: 100 -> 200
  - Added NPU env vars (SGLANG_ENABLE_SPEC_V2, etc.)
  - register_cuda_ci -> register_npu_ci
  - print() -> logger.info()
  - Added --disable-cuda-graph (NPU doesn't support CUDA Graph)
  - Added --sampling-backend ascend
  - TestAdaptiveZeroStepBatchSizeServer NOT ported (depends on GPU routing logic)

Key adaptation from GPU version:
  Same as GPU: create temp config.json with candidate_steps=[1,3], use
  /server_info internal_states to drive upshift/downshift with high/low
  acceptance prompts. The /set_args endpoint is NOT used (GPU version
  also does not use /set_args for adaptive; it uses natural prompts).
"""

import json
import os
import tempfile
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

# High-acceptance prompt: easy to predict, high draft token acceptance
HIGH_ACCEPT_PROMPT = (
    "Output exactly 128 new lines. "
    "Every line must be READY. "
    "Do not add numbering, punctuation, or commentary."
)

# Low-acceptance prompt: creative, low draft token acceptance
LOW_ACCEPT_PROMPT = (
    "Compose a poem in the style of Emily Dickinson about quantum entanglement. "
    "Make it emotionally resonant and at least 100 words."
)

MAX_UPSHIFT_ATTEMPTS = 4
MAX_DOWNSHIFT_ATTEMPTS = 6


class TestNPUAdaptiveSpeculativeServer(CustomTestCase):
    """Test Adaptive Speculative Decoding on NPU.

    Ported from GPU: sgl-project/sglang test/test_adaptive_speculative.py
    Class: TestAdaptiveSpeculativeServer

    This test verifies the adaptive speculative decoding system end-to-end:
    1. Create a config.json with candidate_steps=[1, 3]
    2. Start server with --speculative-adaptive --speculative-adaptive-config
    3. Drive upshift: send high-acceptance prompts, verify num_steps -> 3
    4. Drive downshift: send low-acceptance prompts, verify num_steps -> 1
    5. Drive upshift again
    6. Run GSM8K to verify accuracy is maintained

    This is a faithful port of the GPU test, using the same config.json
    approach and the same high/low acceptance prompts to drive the
    adaptive logic naturally (no /set_args needed).
    """

    model = QWEN3_8B_WEIGHTS_PATH
    draft_model = QWEN3_8B_EAGLE3_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        # Create temp config.json with candidate_steps=[1, 3]
        # This constrains adaptive to only switch between 1 and 3,
        # making the upshift/downshift assertions deterministic.
        #
        # NPU B090 image requires batch-size keyed format:
        #   {"<bs>": {"candidate_steps": [...], ...}}
        # See error: "must contain at least one integer-string BS key,
        #   e.g. {"1": {"candidate_steps": [1,3,7]}}"
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "1": {
                        "candidate_steps": [1, 3],
                        "ema_alpha": 1.0,
                        "warmup_batches": 1,
                        "update_interval": 1,
                        "up_hysteresis": 0.0,
                    }
                },
                f,
            )
            cls.adaptive_config_path = f.name

        logger.info("Created adaptive config at: %s", cls.adaptive_config_path)
        logger.info("Model: %s", cls.model)
        logger.info("Draft model: %s", cls.draft_model)

        try:
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
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
                    "--speculative-adaptive",
                    "--speculative-adaptive-config",
                    cls.adaptive_config_path,
                ],
                env=NPU_ENV,
            )
            logger.info("Adaptive server started successfully.")
        except Exception:
            os.unlink(cls.adaptive_config_path)
            raise

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process"):
            kill_process_tree(cls.process.pid)
        if os.path.exists(cls.adaptive_config_path):
            os.unlink(cls.adaptive_config_path)

    def _get_internal_state(self) -> dict:
        """Get internal state from /server_info.

        Same as GPU version: internal_states[0] contains the adaptive state
        including speculative_num_steps and avg_spec_accept_length.
        """
        response = requests.get(self.base_url + "/server_info", timeout=30)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["internal_states"][0]

    def _generate(self, prompt: str, max_new_tokens: int = 64) -> dict:
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=180,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _drive_upshift(self) -> dict:
        """Send high-acceptance prompts until steps upshift to 3."""
        state = self._get_internal_state()
        for _ in range(MAX_UPSHIFT_ATTEMPTS):
            self._generate(HIGH_ACCEPT_PROMPT)
            state = self._get_internal_state()
            if state["speculative_num_steps"] == 3:
                return state
        return state

    def _drive_downshift(self) -> dict:
        """Send low-acceptance prompts until steps downshift to 1."""
        state = self._get_internal_state()
        for _ in range(MAX_DOWNSHIFT_ATTEMPTS):
            self._generate(LOW_ACCEPT_PROMPT)
            state = self._get_internal_state()
            if state["speculative_num_steps"] == 1:
                return state
        return state

    def test_gsm8k_after_adaptive_switches(self):
        """Exercise up/down/up adaptive switches, then verify GSM8K accuracy.

        This is a faithful port of the GPU test:
        1. Drive upshift: high-acceptance prompts -> num_steps should become 3
        2. Drive downshift: low-acceptance prompts -> num_steps should become 1
        3. Drive upshift again
        4. Run GSM8K to verify accuracy
        """
        logger.info("=== Driving upshift (high-acceptance prompts) ===")
        state = self._drive_upshift()
        self.assertEqual(
            state["speculative_num_steps"], 3, f"Never upshifted: {state}"
        )
        logger.info("Upshifted to num_steps=3: %s", state)

        logger.info("=== Driving downshift (low-acceptance prompts) ===")
        state = self._drive_downshift()
        self.assertEqual(
            state["speculative_num_steps"], 1, f"Never downshifted: {state}"
        )
        logger.info("Downshifted to num_steps=1: %s", state)

        logger.info("=== Driving upshift again ===")
        self._drive_upshift()

        logger.info("=== Running GSM8K ===")
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
        logger.info("GSM8K metrics (adaptive speculative): %s", metrics)

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (Adaptive Speculative on NPU)\n"
                f'{metrics["score"]=:.3f}\n'
            )

        # NPU uses stricter threshold (0.69) than GPU (0.20)
        self.assertGreater(
            metrics["score"],
            0.69,
            "GSM8K score should be > 0.69 with adaptive speculative",
        )

        # Verify avg_spec_accept_length is reported (like GPU version)
        server_info = requests.get(self.base_url + "/server_info").json()
        avg_accept_len = server_info["internal_states"][0]["avg_spec_accept_length"]
        logger.info("avg_spec_accept_length=%.4f", avg_accept_len)


if __name__ == "__main__":
    unittest.main()
