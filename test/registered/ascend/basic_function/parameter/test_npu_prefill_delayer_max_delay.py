"""Manual NPU test for the Prefill Delayer queue-based trigger.

Converted from ``python/sglang/test/prefill.sh``. There are intentionally no
assertions yet: the test only launches a server, drives the same request
pattern as the shell script, and prints the observations (per-request latency
and /metrics gauges) so the effect of the delayer can be eyeballed.

The server is launched with:
    --enable-prefill-delayer
    --prefill-delayer-queue-min-ratio 0.8
    --prefill-delayer-max-delay-ms 5000

Interpretation:
    - If the short requests take close to 5000ms, they were delayed by the
      Prefill Delayer (released on the max-delay timeout).
    - If they finish quickly (<2s), the trigger likely did not fire; try more
      long requests or a larger ratio.
    - Check the server log for "PrefillDelayer" DEBUG lines.
"""

import os
import re
import threading
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# 长请求参数（用于占满 running）
LONG_PROMPT = (
    "Please describe in detail the development history of artificial intelligence "
    "from the Turing test to deep learning, including key figures and milestone "
    "events. Start from the early neural network research in the 1940s, through the "
    "proposal of the Turing test in the 1950s, the birth of the perceptron in the "
    "1960s, the development of expert systems in the 1970s, the breakthrough of the "
    "backpropagation algorithm in the 1980s, the rise of statistical learning in the "
    "1990s, the revival of deep learning in the 2000s, and AlexNet's breakthrough "
    "result in the ImageNet competition in 2012. In recent years, the development of "
    "large language models such as GPT and BERT has pushed artificial intelligence to "
    "new heights."
)
LONG_MAX_TOKENS = 256
LONG_CONCURRENT = 20

# 快速请求参数（用于触发排队）
SHORT_PROMPT = "What is artificial intelligence?"
SHORT_MAX_TOKENS = 50
SHORT_CONCURRENT = 15


class TestNpuPrefillDelayer(CustomTestCase):
    """Testcase: Observe the queue-based Prefill Delayer behavior end-to-end.

    [Test Category] Scheduling
    [Test Target] --enable-prefill-delayer; --prefill-delayer-queue-min-ratio;
                  --prefill-delayer-max-delay-ms
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_0_6B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

        env = os.environ.copy()
        # 调试：开启 PrefillDelayer 的 DEBUG 日志
        env["SGLANG_PREFILL_DELAYER_DEBUG_LOG"] = "1"

        other_args = [
            "--attention-backend",
            "ascend",
            "--enable-metrics",
            "--enable-prefill-delayer",
            "--prefill-delayer-queue-min-ratio",
            "0.8",
            "--prefill-delayer-max-delay-ms",
            "5000",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    @staticmethod
    def _parse_gauge(metrics_text: str, name: str):
        """Return the last value of a prometheus gauge line, or None."""
        matches = re.findall(
            rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)",
            metrics_text,
            re.MULTILINE,
        )
        if not matches:
            return None
        return int(float(matches[-1]))

    def _query_status(self, tag: str):
        try:
            stats = requests.get(f"{self.base_url}/metrics", timeout=10).text
        except Exception as e:  # noqa: BLE001
            print(f"[{tag}] 查询 /metrics 失败: {e}")
            return
        running = self._parse_gauge(stats, "sglang:num_running_reqs")
        waiting = self._parse_gauge(stats, "sglang:num_queue_reqs")
        print(f"[{tag}] num_running_reqs = {running if running is not None else '未知'}")
        print(f"[{tag}] num_queue_reqs   = {waiting if waiting is not None else '未知'}")

    def _send_long_request(self):
        try:
            requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": LONG_PROMPT,
                    "sampling_params": {
                        "max_new_tokens": LONG_MAX_TOKENS,
                        "temperature": 0.7,
                    },
                },
                timeout=120,
            )
        except Exception as e:  # noqa: BLE001
            print(f"长请求异常: {e}")

    def _send_short_request(self, idx: int, results: dict):
        start = time.perf_counter()
        try:
            requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": SHORT_PROMPT,
                    "sampling_params": {
                        "max_new_tokens": SHORT_MAX_TOKENS,
                        "temperature": 0.7,
                    },
                },
                timeout=120,
            )
        except Exception as e:  # noqa: BLE001
            print(f"快速请求 {idx} 异常: {e}")
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        results[idx] = elapsed_ms

    def test_prefill_delayer_stepwise(self):
        print("=== Prefill Delayer 分步测试 ===")
        print(f"目标服务: {self.base_url}")

        # 1. 先发送长请求（后台运行，持续占用 running）
        print(f"[步骤1] 发送 {LONG_CONCURRENT} 个长请求（后台运行，持续占用 running）...")
        long_threads = [
            threading.Thread(target=self._send_long_request)
            for _ in range(LONG_CONCURRENT)
        ]
        for t in long_threads:
            t.start()
        print(f"    已启动 {LONG_CONCURRENT} 个长请求")

        # 等待 1 秒，在长请求 prefill 阶段就发送短请求
        time.sleep(1)

        # 2. 查询当前状态（应看到 running>0, waiting=0）
        print("[步骤2] 查询当前状态（应看到 running>0, waiting=0）")
        self._query_status("步骤2")

        # 3. 在长请求 prefill 阶段发送短请求，记录耗时
        print(f"[步骤3] 在长请求 prefill 阶段发送 {SHORT_CONCURRENT} 个快速请求...")
        print("    预期: running>0, ratio=0.8 → queue_min = min(running*0.8, max_prefill_bs)")
        print(f"    由于 waiting({SHORT_CONCURRENT}) < queue_min，应触发 delay...")
        results = {}
        short_threads = [
            threading.Thread(target=self._send_short_request, args=(i, results))
            for i in range(1, SHORT_CONCURRENT + 1)
        ]
        for t in short_threads:
            t.start()
        for t in short_threads:
            t.join()

        print("快速请求耗时结果：")
        for idx in sorted(results):
            print(f"    {idx}: {results[idx]:.0f}ms")

        # 4. 再次查询状态
        print("[步骤4] 再次查询状态")
        time.sleep(1)
        self._query_status("步骤4")

        # 等待后台长请求结束
        for t in long_threads:
            t.join()

        print("=== 测试完成 ===")
        print("解读：")
        print("   - 快速请求耗时接近 5000ms，说明被 Prefill Delayer 延迟了。")
        print("   - 若耗时很短(<2s)，可能未触发，可增大 running 数或 ratio。")
        print("   - 查看服务日志，应出现 'PrefillDelayer' 相关 DEBUG 信息。")


if __name__ == "__main__":
    unittest.main()
