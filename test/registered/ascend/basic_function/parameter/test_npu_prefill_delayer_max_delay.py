"""Manual NPU test for the Prefill Delayer queue-based trigger.

Converted from ``python/sglang/test/prefill.sh``.

The server is launched with:
    --enable-prefill-delayer
    --prefill-delayer-queue-min-ratio 0.8
    --prefill-delayer-max-delay-ms 5000

Two test classes:
    - TestNpuPrefillDelayer: observation only (no assertions). Drives the same
      request pattern as the shell script and prints per-request latency and
      /metrics gauges so the effect of the delayer can be eyeballed.
    - TestNpuPrefillDelayerBelowThreshold: sends a request count well below the
      queue threshold while keeping running high for the whole delay window, so
      the delay is released only on the max_delay_ms timeout. Asserts the short
      requests wait longer than max_delay_ms.

Interpretation:
    - Delay is released either when the waiting queue reaches
      queue_min = min(running * ratio, max_prefill_bs), or when the wall-clock
      max_delay_ms timeout is hit, whichever comes first.
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

# server 配置
MODEL_PATH = QWEN3_0_6B_WEIGHTS_PATH
MAX_DELAY_MS = 5000
QUEUE_MIN_RATIO = 0.8

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
# 更大的长请求 max_tokens：保证 running 在整个 max_delay 窗口内维持高位，
# 否则长请求过早 decode 完会导致 queue_min 下降、延迟提前放行。
LONG_SUSTAINED_MAX_TOKENS = 1024
LONG_CONCURRENT = 20

# 快速请求参数（用于触发排队）
SHORT_PROMPT = "What is artificial intelligence?"
SHORT_MAX_TOKENS = 50
SHORT_CONCURRENT = 15
# 远小于阈值 queue_min = min(running*0.8, max_prefill_bs) ≈ 16 的请求数，
# 使 waiting 始终 < queue_min，延迟只能靠 max_delay_ms 超时释放。
SHORT_CONCURRENT_BELOW = 3

# 模块级 server 进程，两个测试类共享
GLOBAL_SERVER_PROCESS = None


def setUpModule():
    global GLOBAL_SERVER_PROCESS
    env = os.environ.copy()
    # 调试：开启 PrefillDelayer 的 DEBUG 日志
    env["SGLANG_PREFILL_DELAYER_DEBUG_LOG"] = "1"

    other_args = [
        "--attention-backend",
        "ascend",
        "--enable-metrics",
        "--enable-prefill-delayer",
        "--prefill-delayer-queue-min-ratio",
        str(QUEUE_MIN_RATIO),
        "--prefill-delayer-max-delay-ms",
        str(MAX_DELAY_MS),
    ]
    GLOBAL_SERVER_PROCESS = popen_launch_server(
        MODEL_PATH,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
        env=env,
    )


def tearDownModule():
    if GLOBAL_SERVER_PROCESS is not None:
        kill_process_tree(GLOBAL_SERVER_PROCESS.pid)


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


def _query_status(base_url: str, tag: str):
    try:
        stats = requests.get(f"{base_url}/metrics", timeout=10).text
    except Exception as e:  # noqa: BLE001
        print(f"[{tag}] 查询 /metrics 失败: {e}")
        return None, None
    running = _parse_gauge(stats, "sglang:num_running_reqs")
    waiting = _parse_gauge(stats, "sglang:num_queue_reqs")
    print(f"[{tag}] num_running_reqs = {running if running is not None else '未知'}")
    print(f"[{tag}] num_queue_reqs   = {waiting if waiting is not None else '未知'}")
    return running, waiting


def _send_long_request(base_url: str, max_tokens: int = LONG_MAX_TOKENS):
    try:
        requests.post(
            f"{base_url}/generate",
            json={
                "text": LONG_PROMPT,
                "sampling_params": {
                    "max_new_tokens": max_tokens,
                    "temperature": 0.7,
                },
            },
            timeout=180,
        )
    except Exception as e:  # noqa: BLE001
        print(f"长请求异常: {e}")


def _send_short_request(base_url: str, idx: int, results: dict):
    start = time.perf_counter()
    try:
        requests.post(
            f"{base_url}/generate",
            json={
                "text": SHORT_PROMPT,
                "sampling_params": {
                    "max_new_tokens": SHORT_MAX_TOKENS,
                    "temperature": 0.7,
                },
            },
            timeout=180,
        )
    except Exception as e:  # noqa: BLE001
        print(f"快速请求 {idx} 异常: {e}")
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    results[idx] = elapsed_ms


class TestNpuPrefillDelayer(CustomTestCase):
    """Testcase: Observe the queue-based Prefill Delayer behavior end-to-end.

    [Test Category] Scheduling
    [Test Target] --enable-prefill-delayer; --prefill-delayer-queue-min-ratio;
                  --prefill-delayer-max-delay-ms
    """

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST

    def test_prefill_delayer_stepwise(self):
        print("=== Prefill Delayer 分步测试 ===")
        print(f"目标服务: {self.base_url}")

        # 1. 先发送长请求（后台运行，持续占用 running）
        print(f"[步骤1] 发送 {LONG_CONCURRENT} 个长请求（后台运行，持续占用 running）...")
        long_threads = [
            threading.Thread(target=_send_long_request, args=(self.base_url,))
            for _ in range(LONG_CONCURRENT)
        ]
        for t in long_threads:
            t.start()
        print(f"    已启动 {LONG_CONCURRENT} 个长请求")

        # 等待 1 秒，在长请求 prefill 阶段就发送短请求
        time.sleep(1)

        # 2. 查询当前状态（应看到 running>0, waiting=0）
        print("[步骤2] 查询当前状态（应看到 running>0, waiting=0）")
        _query_status(self.base_url, "步骤2")

        # 3. 在长请求 prefill 阶段发送短请求，记录耗时
        print(f"[步骤3] 在长请求 prefill 阶段发送 {SHORT_CONCURRENT} 个快速请求...")
        print("    预期: running>0, ratio=0.8 → queue_min = min(running*0.8, max_prefill_bs)")
        print(f"    由于 waiting({SHORT_CONCURRENT}) < queue_min，应触发 delay...")
        results = {}
        short_threads = [
            threading.Thread(
                target=_send_short_request, args=(self.base_url, i, results)
            )
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
        _query_status(self.base_url, "步骤4")

        # 等待后台长请求结束
        for t in long_threads:
            t.join()

        print("=== 测试完成 ===")
        print("解读：")
        print("   - 快速请求耗时接近 5000ms，说明被 Prefill Delayer 延迟了。")
        print("   - 若耗时很短(<2s)，可能未触发，可增大 running 数或 ratio。")
        print("   - 查看服务日志，应出现 'PrefillDelayer' 相关 DEBUG 信息。")


class TestNpuPrefillDelayerBelowThreshold(CustomTestCase):
    """Testcase: When the waiting queue stays below the threshold and running is
    kept high for the whole delay window, prefill is released only on the
    max_delay_ms timeout, so the short requests must wait longer than 5s.

    [Test Category] Scheduling
    [Test Target] --prefill-delayer-max-delay-ms
    """

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST

    def test_below_threshold_hits_max_delay(self):
        print("=== 低于阈值请求：应触发 max_delay 超时 ===")
        print(f"目标服务: {self.base_url}")

        # 1. 发送长请求，用更大的 max_tokens 让 running 在 5s 窗口内维持高位
        print(
            f"[步骤1] 发送 {LONG_CONCURRENT} 个长请求"
            f"（max_new_tokens={LONG_SUSTAINED_MAX_TOKENS}，维持 running）..."
        )
        long_threads = [
            threading.Thread(
                target=_send_long_request,
                args=(self.base_url, LONG_SUSTAINED_MAX_TOKENS),
            )
            for _ in range(LONG_CONCURRENT)
        ]
        for t in long_threads:
            t.start()
        print(f"    已启动 {LONG_CONCURRENT} 个长请求")

        # 等待 running 起来（NPU 首次 prefill 有延迟，需比 stepwise 多等一会）
        time.sleep(3)

        # 2. 查询状态，确认 running 已被占满
        print("[步骤2] 查询当前状态（发送短请求前）")
        _query_status(self.base_url, "发送前")

        # 3. 发送远小于阈值的短请求
        print(f"[步骤3] 发送 {SHORT_CONCURRENT_BELOW} 个快速请求（远小于 queue_min≈16）...")
        print(f"    预期: waiting({SHORT_CONCURRENT_BELOW}) 始终 < queue_min，")
        print(f"          延迟只能靠 max_delay_ms={MAX_DELAY_MS} 超时释放 → 耗时 > 5s")
        results = {}
        short_threads = [
            threading.Thread(
                target=_send_short_request, args=(self.base_url, i, results)
            )
            for i in range(1, SHORT_CONCURRENT_BELOW + 1)
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
        _query_status(self.base_url, "释放后")

        # 等待后台长请求结束
        for t in long_threads:
            t.join()

        # 断言：低于阈值的请求应被延迟到 max_delay_ms 超时，耗时 > 5s
        avg_ms = sum(results.values()) / len(results)
        print(f"平均耗时: {avg_ms:.0f}ms（预期 > {MAX_DELAY_MS}ms）")
        self.assertGreater(
            avg_ms,
            MAX_DELAY_MS,
            f"平均耗时 {avg_ms:.0f}ms 未超过 max_delay_ms({MAX_DELAY_MS}ms)，"
            f"延迟可能被提前释放（检查 running 是否维持足够高）",
        )


if __name__ == "__main__":
    unittest.main()
