"""--allowed-media-domains / --media-url-max-file-size-mb: remote media
security gates, verified against a real server.

[Test Category] Parameter
[Test Target] --allowed-media-domains, --media-url-max-file-size-mb

Both cases launch a real multimodal server and send /generate requests whose
image_data points at a local media origin HTTP server, so the gates run on
the actual download path.

--allowed-media-domains
Consumption points: utils/common.py:1511 (configure_media_url_security),
arg_groups/serving_hook.py:189, multimodal/processors/base_processor.py:247,
disaggregation/encoder/server.py:549.
Verified: an image URL on a whitelisted host is downloaded and decoded; a URL
whose hostname is not in the whitelist is rejected with
"Media URL domain is not allowed" (hostnames are normalized, so localhost !=
127.0.0.1); the file:// scheme is rejected.

--media-url-max-file-size-mb
Consumption points: arg_groups/serving_hook.py:191,
multimodal/processors/base_processor.py:249, disaggregation/encoder/server.py:551;
validation at utils/common.py:1523.
Verified: with a 1 MiB cap, a 2 MiB download is rejected mid-stream with
"Remote media exceeds the N byte download limit"; a file under the cap passes.
"""

import unittest

from para_case_common import (
    COMMON_OTHER_ARGS,
    HYBRID,
    AscendServer,
    AscendServerTestCase,
    media_http_server,
    server_info,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=3600, suite="nightly-1-npu-a3", nightly=True)


class TestAllowedMediaDomains(AscendServerTestCase):
    PORT = 31410

    @classmethod
    def setUpClass(cls):
        cls.origin, cls.origin_port = media_http_server()
        cls.server = AscendServer(
            "21_allowed_media_domains",
            HYBRID,
            cls.PORT,
            other_args=COMMON_OTHER_ARGS + ["--allowed-media-domains", "127.0.0.1"],
            timeout=1800,
        ).start()
        cls.server.join_launch()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        cls.origin.shutdown()

    def _post(self, image_url):
        from para_case_common import http

        payload = {
            "text": "describe the image",
            "image_data": [image_url],
            "sampling_params": {"max_new_tokens": 8, "temperature": 0},
        }
        return http(self.PORT, "/generate", method="post", payload=payload, timeout=120)

    def test_whitelisted_image_url_accepted(self):
        """An image URL on the whitelisted host is downloaded and decoded."""
        info = server_info(self.PORT)
        self.assertEqual(info["allowed_media_domains"], ["127.0.0.1"])

        r = self._post(f"http://127.0.0.1:{self.origin_port}/small")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json()["text"])

    def test_non_whitelisted_host_and_scheme_rejected(self):
        """localhost is not whitelisted (hostname normalization) and file:// is invalid."""
        r = self._post(f"http://localhost:{self.origin_port}/small")
        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("Media URL domain is not allowed", r.text)

        r = self._post("file:///etc/passwd")
        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("Invalid media URL", r.text)


class TestMediaUrlMaxFileSizeMb(AscendServerTestCase):
    PORT = 31420

    @classmethod
    def setUpClass(cls):
        cls.origin, cls.origin_port = media_http_server()
        cls.server = AscendServer(
            "22_media_url_max_file_size_mb",
            HYBRID,
            cls.PORT,
            other_args=COMMON_OTHER_ARGS + ["--media-url-max-file-size-mb", "1"],
            timeout=1800,
        ).start()
        cls.server.join_launch()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        cls.origin.shutdown()

    def _post(self, image_url):
        from para_case_common import http

        payload = {
            "text": "describe the image",
            "image_data": [image_url],
            "sampling_params": {"max_new_tokens": 8, "temperature": 0},
        }
        return http(self.PORT, "/generate", method="post", payload=payload, timeout=120)

    def test_download_size_cap_enforced(self):
        """1 MiB cap: a 2 MiB download is rejected mid-stream, a small one passes."""
        info = server_info(self.PORT)
        self.assertEqual(info["media_url_max_file_size_mb"], 1)

        # Under the cap: valid 1x1 PNG downloads and decodes fine
        r = self._post(f"http://127.0.0.1:{self.origin_port}/small")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json()["text"])

        # Over the cap: 2 MiB body rejected during the streaming download
        r = self._post(f"http://127.0.0.1:{self.origin_port}/big")
        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("exceeds the", r.text)
        self.assertIn("byte download limit", r.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
