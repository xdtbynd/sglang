"""--mamba-radix-cache-strategy: Mamba radix cache strategy (auto/no_buffer/extra_buffer/extra_buffer_lazy).

Verification level: confirmed
Consumption points: arg_groups/model_override_base.py:289 (decides enable_mamba_extra_buffer),
arg_groups/mamba_hook.py:110
Server launch style: sglang/test/ascend (popen_launch_server + kill_process_tree).
Evidence: the consumption chain is complete and real; on hybrid linear models the log shows
`Mamba Cache is allocated. max_mamba_cache_size`.
Note: under this case's config (--page-size 1 --disable-overlap-schedule), auto also
derives no_buffer, so the explicit setting is equivalent to the default with no
observable difference.

[Test Category] Parameter
[Test Target] --mamba-radix-cache-strategy
"""

import unittest

from para_case_common import (
    COMMON_OTHER_ARGS,
    HYBRID,
    AscendServer,
    AscendServerTestCase,
    generate,
    server_info,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1800, suite="nightly-1-npu-a3", nightly=True)


class TestMambaRadixCacheStrategy(AscendServerTestCase):
    PORT = 31220

    @classmethod
    def setUpClass(cls):
        cls.server = AscendServer(
            "09_mamba_radix_cache_strategy",
            HYBRID,
            cls.PORT,
            other_args=COMMON_OTHER_ARGS
            + [
                "--mamba-radix-cache-strategy",
                "no_buffer",
                "--page-size",
                "1",
                "--disable-overlap-schedule",
            ],
            timeout=1800,
        ).start()
        cls.server.join_launch()

    def test_mamba_radix_cache_strategy_no_buffer(self):
        """--mamba-radix-cache-strategy=no_buffer: the Mamba pool is allocated with no_buffer."""
        info = server_info(self.PORT)
        self.assertEqual(info["mamba_radix_cache_strategy"], "no_buffer")
        self.assertTrue(info["uses_mamba_radix_cache"])
        log = self.server.log()
        self.assertIn("Mamba Cache is allocated. max_mamba_cache_size", log)
        out = generate(self.PORT, "mamba cache probe ->")
        self.assertTrue(out["text"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
