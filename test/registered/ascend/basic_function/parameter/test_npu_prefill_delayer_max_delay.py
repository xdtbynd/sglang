"""NPU test for the Prefill Delayer queue-based trigger: above/below threshold.

Converted from ``python/sglang/test/prefill.sh``.

The server is launched with:
    --enable-prefill-delayer
    --prefill-delayer-queue-min-ratio 0.8
    --prefill-delayer-max-delay-ms {MAX_DELAY_MS}

Two test classes (both with assertions):
    - TestNpuPrefillDelayerAboveThreshold: sends requests exceeding
      queue_min (= min(running*ratio, max_prefill_bs)) so the condition
      "waiting >= queue_min" is met and prefill is released promptly.
      Asserts short requests complete below max_delay_ms.
    - TestNpuPrefillDelayerBelowThreshold: sends a request count far below
      queue_min while keeping running high. The delay can only be released
      on the max_delay_ms timeout. Asserts each short request waits > max_delay_ms.

Interpretation:
    - queue_min = min(running * ratio, max_prefill_bs).
    - waiting >= queue_min → no delay, immediate release.
    - waiting < queue_min → delay until queue fills or max_delay_ms timeout.
    - Check server log for "PrefillDelayer" DEBUG lines.
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
# 2048 token 的 decode 窗口（~2300 tok/s）约十几秒，足以覆盖短请求的 5s 延迟。
LONG_SUSTAINED_MAX_TOKENS = 2048
LONG_CONCURRENT = 20

# 快速请求参数（用于触发排队）
SHORT_PROMPT = "What is artificial intelligence?"
SHORT_MAX_TOKENS = 50
# 超过阈值：queue_min = min(running*0.8, max_prefill_bs) ≈ 16 (running=20时)，
# 发 25 远大于 queue_min → waiting >= queue_min → 无延迟，立刻下发。
SHORT_CONCURRENT_ABOVE = 25
# 远小于阈值 → waiting 始终 < queue_min，延迟只能靠 max_delay_ms 超时释放。
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


def _query_status(base_url: str, tag: str, verbose: bool = True):
    try:
        stats = requests.get(f"{base_url}/metrics", timeout=10).text
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"[{tag}] 查询 /metrics 失败: {e}")
        return None, None
    running = _parse_gauge(stats, "sglang:num_running_reqs")
    waiting = _parse_gauge(stats, "sglang:num_queue_reqs")
    if verbose:
        print(f"[{tag}] num_running_reqs = {running if running is not None else '未知'}")
        print(f"[{tag}] num_queue_reqs   = {waiting if waiting is not None else '未知'}")
    return running, waiting


def _wait_until_stable_running(
    base_url: str,
    expected_running: int,
    tolerance: int = 2,
    stable_secs: float = 4.0,
    interval: float = 0.5,
    timeout: float = 40.0,
):
    """轮询 /metrics，直到 running 连续 stable_secs 秒维持 >= 目标水位。

    目标水位 = expected_running - tolerance，容忍个别长请求的偶发波动
    （如提前结束/调度抖动），否则一旦 running 从 20 掉到 19 就永远达不到
    "连续 4s == 20"而超时。

    仅凭单次达标不够：metrics 计数早于实际 prefill/decode，running 在长请求
    真正 prefill 前就已显示为高位。用"持续稳定"跨过长请求自身的 prefill/flush
    延迟阶段，确保短请求发出时系统已进入持续 decode。返回稳定时的 running，
    超时返回 None。
    """
    target = max(1, expected_running - tolerance)
    deadline = time.perf_counter() + timeout
    stable_since = None
    while time.perf_counter() < deadline:
        running, _ = _query_status(base_url, "wait", verbose=False)
        now = time.perf_counter()
        if running is not None and running >= target:
            if stable_since is None:
                stable_since = now
            elif now - stable_since >= stable_secs:
                return running
        else:
            stable_since = None
        time.sleep(interval)
    return None


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
                # 必须跑满 max_tokens：否则长请求可能提前 EOS 结束，running
                # 掉下来，_wait_until_stable_running 无法维持稳定而超时。
                "ignore_eos": True,
            },
            timeout=180,
        )
    except Exception as e:  # noqa: BLE001
        print(f"长请求异常: {e}")


def _send_short_request(base_url: str, idx: int, results: dict):
    """发送短请求，将 (elapsed_ms, status_code) 写入 results[idx]。"""
    start = time.perf_counter()
    status = None
    try:
        resp = requests.post(
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
        status = resp.status_code
    except Exception as e:  # noqa: BLE001
        print(f"快速请求 {idx} 异常: {e}")
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    results[idx] = (elapsed_ms, status)


def _print_short_results(results: dict):
    """打印短请求耗时结果，同时检查全部 status_code=200。"""
    all_ok = True
    for idx in sorted(results):
        elapsed_ms, status = results[idx]
        status_str = "OK" if status == 200 else str(status)
        print(f"    {idx}: {elapsed_ms:.0f}ms  (HTTP {status_str})")
        if status != 200:
            all_ok = False
    return all_ok


class TestNpuPrefillDelayerAboveThreshold(CustomTestCase):
    """Testcase: Send short requests exceeding queue_min so the condition
    "waiting >= queue_min" is met and prefill is released promptly (no delay).

    [Test Category] Scheduling
    [Test Target] --enable-prefill-delayer; --prefill-delayer-queue-min-ratio;
                  --prefill-delayer-max-delay-ms
    """

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST

    def test_above_threshold_released_immediately(self):
        print("=== 超过阈值请求：应立刻下发，无延迟 ===")
        print(f"目标服务: {self.base_url}")

        # 1. 发送长请求占满 running
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

        # 2. 轮询等到 stable running
        print("[步骤2] 轮询等待长请求进入稳定 decode...")
        stable_running = _wait_until_stable_running(
            self.base_url, expected_running=LONG_CONCURRENT
        )
        self.assertIsNotNone(
            stable_running,
            f"长请求未能稳定占满 running(>= {LONG_CONCURRENT})",
        )
        running, waiting = _query_status(self.base_url, "发送前")

        # 3. 发送超过阈值的短请求
        queue_min_est = int(stable_running * QUEUE_MIN_RATIO) if stable_running else "?"
        print(
            f"[步骤3] 发送 {SHORT_CONCURRENT_ABOVE} 个快速请求"
            f"（超过 queue_min≈{queue_min_est}）..."
        )
        print(
            f"    预期: waiting({SHORT_CONCURRENT_ABOVE}) >= queue_min"
            f" → 立刻下发，耗时 < {MAX_DELAY_MS}ms"
        )
        results = {}
        short_threads = [
            threading.Thread(
                target=_send_short_request, args=(self.base_url, i, results)
            )
            for i in range(1, SHORT_CONCURRENT_ABOVE + 1)
        ]
        for t in short_threads:
            t.start()
        for t in short_threads:
            t.join()

        print("快速请求耗时结果：")
        all_ok = _print_short_results(results)

        # 4. 再次查询状态
        print("[步骤4] 再次查询状态")
        _query_status(self.base_url, "释放后")

        # 等待后台长请求结束
        for t in long_threads:
            t.join()

        # 断言
        self.assertTrue(all_ok, "部分短请求未返回 HTTP 200")
        elapsed_values = [ms for ms, _ in results.values()]
        avg_ms = sum(elapsed_values) / len(elapsed_values)
        print(f"平均耗时: {avg_ms:.0f}ms（预期 < {MAX_DELAY_MS}ms）")
        self.assertLess(
            avg_ms,
            MAX_DELAY_MS,
            f"平均耗时 {avg_ms:.0f}ms 未低于 max_delay_ms({MAX_DELAY_MS}ms)，"
            f"超过阈值后仍被延迟，不符合预期",
        )


class TestNpuPrefillDelayerBelowThreshold(CustomTestCase):
    """Testcase: When the waiting queue stays below the threshold and running is
    kept high for the whole delay window, prefill is released only on the
    max_delay_ms timeout, so the short requests must wait longer than
    max_delay_ms.

    [Test Category] Scheduling
    [Test Target] --prefill-delayer-max-delay-ms
    """

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST

    def test_below_threshold_hits_max_delay(self):
        print("=== 低于阈值请求：应触发 max_delay 超时 ===")
        print(f"目标服务: {self.base_url}")

        # 1. 发送长请求，用更大的 max_tokens 让 running 在整个延迟窗口内维持高位
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

        # 2. 轮询直到长请求进入稳定 decode（running 连续维持高位），而非固定
        #    sleep：确保短请求落在干净的稳定窗口，不被长请求自身的
        #    prefill/flush 延迟周期裹挟。
        print("[步骤2] 轮询等待长请求进入稳定 decode...")
        stable_running = _wait_until_stable_running(
            self.base_url, expected_running=LONG_CONCURRENT
        )
        self.assertIsNotNone(
            stable_running,
            f"长请求未能稳定占满 running(>= {LONG_CONCURRENT})，无法验证 max_delay",
        )
        _query_status(self.base_url, "发送前")

        # 3. 发送远小于阈值的短请求
        print(f"[步骤3] 发送 {SHORT_CONCURRENT_BELOW} 个快速请求（远小于 queue_min≈16）...")
        print(f"    预期: waiting({SHORT_CONCURRENT_BELOW}) 始终 < queue_min，")
        print(
            f"          延迟只能靠 max_delay_ms={MAX_DELAY_MS} 超时释放"
            f" → 耗时 > {MAX_DELAY_MS}ms"
        )
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
        all_ok = _print_short_results(results)

        # 4. 再次查询状态
        print("[步骤4] 再次查询状态")
        _query_status(self.base_url, "释放后")

        # 等待后台长请求结束
        for t in long_threads:
            t.join()

        self.assertTrue(all_ok, "部分短请求未返回 HTTP 200")
        # 断言：低于阈值的每个请求都应被延迟到 max_delay_ms 超时
        for idx in sorted(results):
            elapsed_ms, _ = results[idx]
            self.assertGreater(
                elapsed_ms,
                MAX_DELAY_MS,
                f"请求 {idx} 耗时 {elapsed_ms:.0f}ms 未超过"
                f" max_delay_ms({MAX_DELAY_MS}ms)，"
                f"延迟可能被提前释放",
            )


if __name__ == "__main__":
    unittest.main()
