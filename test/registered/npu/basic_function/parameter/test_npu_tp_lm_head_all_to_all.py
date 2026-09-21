"""--enable-tp-lm-head-all-to-all: under DP attention, the LM head uses all-to-all instead of all-gather+scatter.

Verification level: confirmed (all-to-all actually executed)
Consumption points: layers/logits_processor.py:415 → :848 →
:1004 _can_use_tp_lm_head_all_to_all → :1043 _tp_lm_head_all_to_all
Preconditions (three gates, logits_processor.py:1004-1041):
  1. lm_head must be TP-sharded (base_lm_head.tp_size == tp_size)
     → equivalent to requiring tie_word_embeddings=false; tied models always fall back
  2. The logprob token counts of all dp ranks must be equal (global_num_tokens_for_logprob_cpu)
     → a single request is always unequal under dp2; equal-length concurrency is required
  3. Number of logits rows == local rows * tp_size
Server launch style: sglang/test/ascend (popen_launch_server + kill_process_tree),
cards=CARDS_TP2 (two-card / tp2).
Evidence (per-frame trace with SGLANG_TRACE_LOGITS_E2E=1): all-to-all frames have
tp_logits_gather_returned but no dp_logits_scatter; fallback frames have both.
Model: Qwen3.5-9B (natively non-tied and already adapted for dp-attention).

[Test Category] Parameter
[Test Target] --enable-tp-lm-head-all-to-all
"""

import concurrent.futures
import re
import unittest

from para_case_common import (
    A2A_EXTRA_ARGS,
    A2A_MODEL,
    CARDS_TP2,
    COMMON_OTHER_ARGS,
    AscendServer,
    AscendServerTestCase,
    generate,
    server_info,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1800, suite="nightly-1-npu-a3", nightly=True)


class TestTpLmHeadAllToAll(AscendServerTestCase):
    PORT = 31300

    @classmethod
    def setUpClass(cls):
        cls.server = AscendServer(
            "17_tp_lm_head_a2a",
            A2A_MODEL,
            cls.PORT,
            other_args=COMMON_OTHER_ARGS
            + [
                "--tp-size",
                "2",
                "--dp-size",
                "2",
                "--enable-dp-attention",
                "--enable-tp-lm-head-all-to-all",
            ]
            + A2A_EXTRA_ARGS,
            cards=CARDS_TP2,
            # Enable the logits communication path trace to determine whether the a2a
            # branch really executed
            # (logits_processor.py:810-868, prints "SGLANG_TRACE_LOGITS_E2E ... stage=...")
            env_extra={"SGLANG_TRACE_LOGITS_E2E": "1"},
            timeout=1800,
        ).start()
        cls.server.join_launch()

    def _trace_frames_since(self, offset):
        """Collect the logits trace appended after the log offset, split by frame (one get_logits call).

        Each frame records the sequence of stages observed, the token count of each dp
        rank in that frame, and whether dp_logits_scatter appeared. See
        logits_processor.py:841-868 for the criteria:
          a2a path            -> _tp_lm_head_all_to_all, used_tp_lm_head_all_to_all=True,
                                 so the dp scatter is skipped, output shape
                                 (local rows, full vocab)
          all-gather fallback -> after _logits_gatherer the dp scatter is still needed,
                                 output (global rows, full vocab)
        Therefore "tp_logits_gather_returned present and dp_logits_scatter absent"
        means the frame went through a2a.
        """
        frames, cur = [], None
        for line in self.server.log()[offset:].splitlines():
            if "SGLANG_TRACE_LOGITS_E2E" not in line or "stage=" not in line:
                continue
            stage = line.split("stage=", 1)[1].split()[0]
            if stage == "get_logits_enter":
                if cur:
                    frames.append(cur)
                cur = {"stages": [], "counts": None, "scatter": False}
            if cur is None:
                continue
            cur["stages"].append(stage)
            if stage == "dp_metadata_enter":
                m = re.search(r"global_counts_cpu=(\[[^\]]*\])", line)
                if m:
                    cur["counts"] = m.group(1)
            if stage == "dp_logits_scatter_enter":
                cur["scatter"] = True
        if cur:
            frames.append(cur)
        return frames

    @staticmethod
    def _verdict(frames):
        """Per-frame verdict; returns (a2a frame count, fallback frame count, whether the tp gather block was ever entered)."""
        entered = any("tp_logits_gather_enter" in f["stages"] for f in frames)
        a2a = sum(
            1
            for f in frames
            if "tp_logits_gather_returned" in f["stages"] and not f["scatter"]
        )
        fallback = sum(
            1
            for f in frames
            if "tp_logits_gather_returned" in f["stages"] and f["scatter"]
        )
        return a2a, fallback, entered

    def test_tp_lm_head_all_to_all_real_execution(self):
        """Single request falls back (unequal token counts) → 4-way equal-length concurrency really goes through a2a."""
        info = server_info(self.PORT)
        self.assertTrue(info["enable_tp_lm_head_all_to_all"])
        self.assertTrue(info["enable_dp_attention"])
        self.assertEqual(info["tp_size"], 2)
        self.assertEqual(info["dp_size"], 2)

        # Phase 1: single request (the two ranks always have unequal token counts under dp2)
        off1 = len(self.server.log())
        out = generate(self.PORT, "Explain what a tensor is in one line.")
        self.assertTrue(out["text"])
        frames1 = self._trace_frames_since(off1)

        # Phase 2: 4 equal-length concurrent requests (the two dp ranks each get an equal share of tokens)
        off2 = len(self.server.log())
        prompt = "The capital of France is"
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(generate, self.PORT, prompt, 8) for _ in range(4)]
            for f in futs:
                self.assertTrue(f.result()["text"])
        frames2 = self._trace_frames_since(off2)

        a2a1, fb1, entered1 = self._verdict(frames1)
        a2a2, fb2, entered2 = self._verdict(frames2)
        log = self.server.log()
        self.assertTrue(
            entered1 or entered2,
            "Never entered the TP logits gather block; the trace criteria failed; log tail:\n"
            + log[-2000:],
        )
        self.assertGreater(
            a2a2,
            0,
            f"No frame went through all-to-all under 4-way equal-length concurrency (a2a={a2a2}, fallback={fb2})",
        )
        self.assertEqual(
            a2a1,
            0,
            f"Single request has unequal token counts and should fall back entirely (a2a={a2a1}, fallback={fb1})",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
