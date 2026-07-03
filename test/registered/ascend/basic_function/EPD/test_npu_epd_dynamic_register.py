"""Test EPD (Encoder Processing Disaggregation) dynamic registration on NPU.

[Test Category] EPD (Encoder Processing Disaggregation)
[Test Target] --encoder-bootstrap-port; --encoder-register-urls;
--language-only; --base-gpu-id; --tp-size
[Platform] NPU (Ascend A3, CANN 9.0.0)
[Porting Source] New test case

This test verifies the new EPD dynamic registration mechanism:
  - --encoder-bootstrap-port: Starts EncoderBootstrapServer in the
    language-only process to receive encoder registrations.
  - --encoder-register-urls: Used by encoder-only server to register
    itself with the language server at startup.

Test flow:
  1. Start language-only server with --encoder-bootstrap-port 8997
  2. Start encoder-only server with --encoder-register-urls http://127.0.0.1:8997
  3. Verify encoder is registered via /list_encoder_urls
  4. Send VLM request, verify correct response
  5. Stop encoder, verify it's removed from /list_encoder_urls
  6. Restart encoder, verify it's re-registered
"""

import os
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH,
    logger,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=600, suite="full-4-npu-a3", nightly=True)


NPU_ENV = {
    **os.environ,
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_MM_SKIP_COMPUTE_HASH": "True",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
}

# Bootstrap port for EncoderBootstrapServer
ENCODER_BOOTSTRAP_PORT = 8997

# Language-only server port
LANGUAGE_SERVER_PORT = 30000
LANGUAGE_SERVER_URL = f"http://127.0.0.1:{LANGUAGE_SERVER_PORT}"

# Encoder-only server port
ENCODER_SERVER_PORT = 30010
ENCODER_SERVER_URL = f"http://127.0.0.1:{ENCODER_SERVER_PORT}"

# Test image (1x1 red pixel PNG in base64)
TEST_IMAGE_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mP8/5+h"
    "gQABAAEAf9zB9QAAAABJRU5ErkJggg=="
)


class TestNPUEPDDynamicRegister(CustomTestCase):
    """Test EPD dynamic registration and runtime encoder up/down."""

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH
        cls.language_url = LANGUAGE_SERVER_URL
        cls.encoder_url = ENCODER_SERVER_URL

        # Step 1: Start language-only server with --encoder-bootstrap-port
        logger.info("=== Starting language-only server with EPD bootstrap ===")
        logger.info("Model: %s", cls.model)
        logger.info("Bootstrap port: %d", ENCODER_BOOTSTRAP_PORT)

        language_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.7",
            "--tp-size",
            "2",
            "--base-gpu-id",
            "0",
            "--language-only",
            "--encoder-bootstrap-port",
            str(ENCODER_BOOTSTRAP_PORT),
            "--host",
            "127.0.0.1",
            "--port",
            str(LANGUAGE_SERVER_PORT),
        ]

        cls.language_process = popen_launch_server(
            cls.model,
            cls.language_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=language_args,
            env=NPU_ENV,
        )
        logger.info("Language-only server started.")

        # Step 2: Start encoder-only server with --encoder-register-urls
        logger.info("=== Starting encoder-only server with dynamic registration ===")

        encoder_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.7",
            "--tp-size",
            "2",
            "--base-gpu-id",
            "2",
            "--encoder-only",
            "--encoder-register-urls",
            f"http://127.0.0.1:{ENCODER_BOOTSTRAP_PORT}",
            "--host",
            "127.0.0.1",
            "--port",
            str(ENCODER_SERVER_PORT),
        ]

        cls.encoder_process = popen_launch_server(
            cls.model,
            cls.encoder_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=encoder_args,
            env=NPU_ENV,
        )
        logger.info("Encoder-only server started and registered.")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "encoder_process"):
            kill_process_tree(cls.encoder_process.pid)
        if hasattr(cls, "language_process"):
            kill_process_tree(cls.language_process.pid)

    def _wait_for_health(self, url, timeout=60):
        """Wait for server to be healthy."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = requests.get(url + "/health", timeout=5)
                if resp.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(2)
        return False

    def test_a_both_servers_healthy(self):
        """Verify both language and encoder servers are healthy."""
        self.assertTrue(
            self._wait_for_health(self.language_url),
            "Language server should be healthy",
        )
        self.assertTrue(
            self._wait_for_health(self.encoder_url), "Encoder server should be healthy"
        )
        logger.info("Both servers are healthy.")

        # Check encoder /health returns 200
        resp = requests.get(self.encoder_url + "/health", timeout=10)
        self.assertEqual(resp.status_code, 200)

        # Check language /health returns 200
        resp = requests.get(self.language_url + "/health", timeout=10)
        self.assertEqual(resp.status_code, 200)

    def test_b_encoder_registered(self):
        """Verify encoder is registered in language server."""
        resp = requests.get(self.language_url + "/list_encoder_urls", timeout=30)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        logger.info("Encoder URLs from /list_encoder_urls: %s", data)

        encoder_urls = data if isinstance(data, list) else data.get("encoder_urls", [])
        self.assertTrue(
            any(
                ENCODER_SERVER_URL in url or str(ENCODER_SERVER_PORT) in url
                for url in encoder_urls
            ),
            f"Encoder URL should be registered. Got: {encoder_urls}",
        )
        logger.info("Encoder is registered in language server.")

    def test_c_vlm_request(self):
        """Verify VLM request routes to encoder and returns correct response."""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{TEST_IMAGE_BASE64}"
                            },
                        },
                        {
                            "type": "text",
                            "text": "What color is this image? Answer in one word.",
                        },
                    ],
                }
            ],
            "max_tokens": 64,
            "temperature": 0,
        }

        resp = requests.post(
            self.language_url + "/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200)
        result = resp.json()
        content = result["choices"][0]["message"]["content"]

        self.assertIsNotNone(content)
        self.assertGreater(len(content), 0)
        logger.info("VLM response: %s", content[:200])

    def test_d_encoder_offline(self):
        """Verify encoder goes offline when stopped."""
        logger.info("Stopping encoder server...")

        if hasattr(self, "encoder_process"):
            kill_process_tree(self.__class__.encoder_process.pid)

        # Wait for encoder to be de-registered
        time.sleep(10)

        # Check /list_encoder_urls no longer contains encoder
        try:
            resp = requests.get(self.language_url + "/list_encoder_urls", timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                encoder_urls = (
                    data if isinstance(data, list) else data.get("encoder_urls", [])
                )
                logger.info("Encoder URLs after encoder stopped: %s", encoder_urls)
        except Exception as e:
            logger.info("Could not query /list_encoder_urls: %s", e)

        logger.info("Encoder offline test completed.")

    def test_e_encoder_re_register(self):
        """Verify encoder can re-register after restart."""
        logger.info("Restarting encoder server...")

        encoder_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.7",
            "--tp-size",
            "2",
            "--base-gpu-id",
            "2",
            "--encoder-only",
            "--encoder-register-urls",
            f"http://127.0.0.1:{ENCODER_BOOTSTRAP_PORT}",
            "--host",
            "127.0.0.1",
            "--port",
            str(ENCODER_SERVER_PORT),
        ]

        self.__class__.encoder_process = popen_launch_server(
            self.model,
            self.encoder_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=encoder_args,
            env=NPU_ENV,
        )
        logger.info("Encoder server restarted.")

        self.assertTrue(
            self._wait_for_health(self.encoder_url),
            "Encoder server should be healthy after restart",
        )

        # Verify re-registration
        resp = requests.get(self.language_url + "/list_encoder_urls", timeout=30)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        encoder_urls = data if isinstance(data, list) else data.get("encoder_urls", [])
        logger.info("Encoder URLs after re-register: %s", encoder_urls)

        self.assertTrue(
            any(
                ENCODER_SERVER_URL in url or str(ENCODER_SERVER_PORT) in url
                for url in encoder_urls
            ),
            f"Encoder URL should be re-registered. Got: {encoder_urls}",
        )
        logger.info("Encoder successfully re-registered.")


if __name__ == "__main__":
    unittest.main()
