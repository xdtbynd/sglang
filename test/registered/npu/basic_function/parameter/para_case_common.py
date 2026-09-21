"""para_case_common — shared utilities for NPU parameter end-to-end cases
(test/registered/npu/basic_function/parameter).

Server launch follows the sglang/test/ascend test style.

The server launch approach matches python/sglang/test/ascend/gsm8k_ascend_mixin.py
(GSM8KAscendMixin):

    cls.process = popen_launch_server(          # sglang.test.test_utils
        cls.model, cls.base_url, timeout=..., other_args=cls.other_args, env=cls.env
    )
    cls.server_cmd = subprocess.list2cmdline(cls.process.args)
    ...                                         # tearDownClass
    kill_process_tree(cls.process.pid)          # sglang.srt.utils

device is appended by popen_launch_server's auto detection (returns npu on this
machine); the base arguments COMMON_OTHER_ARGS match the mixin. There are only
two differences from the mixin (parameter verification requires observing logs
and special timings):
 1. Server stdout/stderr are written to disk at logs/e2e_<name>.log via
    return_stdout_stderr, for log marker assertions;
 2. popen_launch_server runs in a background thread (internally polling
    /health_generate until ready); each of the two special timings has its own
    dedicated launch method (do not use start() plus manual waiting directly):
      - start_gated(gate_port): --gated-launch-port, server launch blocks at
        the gate, then activate_gate(gate_port) releases it and joins;
      - start_smg_grpc(): --smg-grpc-mode, the main port is gRPC and the sidecar
        (port+1) has no /health_generate; readiness is determined by log markers
        (no join; tearDown kills the process tree).

Directory convention: each test_*.py corresponds to one parameter in para.txt
(numbering matches the para.txt line numbers), and can be run standalone:

  cd test/registered/npu/basic_function/parameter
  python3 test_npu_smg_grpc_mode.py                    # single parameter
  python3 -m pytest test_npu_mamba_radix_cache_strategy.py -x -q

Or run everything:
  python3 -m unittest discover -p "test_*.py" -v
"""

import os
import socket
import subprocess
import threading
import time
import unittest

from sglang.test.test_utils import CustomTestCase

HERE = os.path.dirname(os.path.abspath(__file__))
CASES_LOGDIR = os.environ.get("PARA_LOG_DIR", os.path.join(HERE, "logs"))

# --------------------------------------------------------------------------- #
# Models / card IDs / common launch arguments (overridable via env vars)
# --------------------------------------------------------------------------- #

SMALL = os.environ.get("PARA_SMALL_MODEL", "/home/weights/Qwen3-0.6B")
# Small model with hybrid linear (GDN) + multimodal + built-in MTP head
HYBRID = os.environ.get("PARA_HYBRID_MODEL", "/mnt/paas/weights/Qwen3.5-4B")

# Dedicated to parameter 17: a2a requires lm_head to be TP-sharded
# (tie_word_embeddings=false). Qwen3.5-9B is a natively non-tied GDN hybrid
# model, the only checkpoint on this machine that can genuinely trigger a2a.
A2A_MODEL = os.environ.get("PARA_A2A_MODEL", "/home/weights/Qwen3.5-9B")
A2A_EXTRA_ARGS = os.environ.get(
    "PARA_A2A_EXTRA_ARGS",
    "--mamba-radix-cache-strategy no_buffer --page-size 1 "
    "--disable-overlap-schedule --linear-attn-backend triton",
).split()

# Physical NPUs used by single-card / dual-card (tp2)
CARD = os.environ.get("PARA_CARD", "4")
CARDS_TP2 = os.environ.get("PARA_CARDS_TP2", "1,4")

# Base arguments matching GSM8KAscendMixin.other_args; --device npu is
# appended automatically by popen_launch_server(device="auto").
COMMON_OTHER_ARGS = [
    "--trust-remote-code",
    "--mem-fraction-static",
    os.environ.get("PARA_MEM_FRACTION", "0.8"),
    "--attention-backend",
    "ascend",
    "--dtype",
    "bfloat16",
    "--disable-cuda-graph",
]


# --------------------------------------------------------------------------- #
# Observation / assertion helpers
# --------------------------------------------------------------------------- #


def port_free(port, host="127.0.0.1", timeout=2):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        return s.connect_ex((host, port)) != 0
    finally:
        s.close()


def port_listening(port, host="127.0.0.1", timeout=2):
    return not port_free(port, host, timeout)


def http(port, path, method="get", payload=None, timeout=120):
    import requests

    url = f"http://127.0.0.1:{port}{path}"
    if method == "get":
        return requests.get(url, timeout=timeout)
    return requests.post(url, json=payload, timeout=timeout)


def generate(port, text, max_new_tokens=8, extra=None, timeout=180):
    import requests

    payload = {
        "text": text,
        "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0},
    }
    if extra:
        payload.update(extra)
    r = requests.post(
        f"http://127.0.0.1:{port}/generate", json=payload, timeout=timeout
    )
    r.raise_for_status()
    return r.json()


def server_info(port):
    r = http(port, "/server_info")
    r.raise_for_status()
    return r.json()


# Shared by the media-security cases: local media origin HTTP server
# (/small returns a valid 1x1 PNG, /big returns 2MiB of raw bytes)
def media_http_server():
    """Start a local HTTP media origin; returns (server, port), caller must
    call server.shutdown()."""
    import base64
    import http.server
    import threading

    payload = b"x" * (2 * 1024 * 1024)
    # Valid 1x1 red PNG so a real server can decode it after download
    png_1x1 = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
        "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/big":
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            elif self.path == "/small":
                body = png_1x1
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


# --------------------------------------------------------------------------- #
# Server wrapper in the ascend test style
# --------------------------------------------------------------------------- #


class AscendServer:
    """Server launch in the GSM8KAscendMixin style: popen_launch_server +
    kill_process_tree.

    popen_launch_server runs in a background thread (internally polling
    /health_generate until ready or failing as expected); on top of that this
    class adds:
      - join_launch(): wait until ready (or re-raise the launch failure
        exception of negative cases as-is);
      - start_gated() / activate_gate() / start_smg_grpc(): dedicated launch
        methods for parameters with special timings (smg-grpc-mode and gated
        launch do not go through the health wait);
      - wait_log() / wait_port(): wait for log markers / port listening
        (reused by the methods above and by timing assertions inside cases);
      - stop(): tearDown finishing via kill_process_tree; if
        popen_launch_server has not returned yet (in log-wait mode the process
        object lives inside it), fall back to killing the whole child process
        tree of this process.
    """

    def __init__(
        self,
        name,
        model,
        port,
        other_args=None,
        cards=None,
        env_extra=None,
        timeout=1800,
    ):
        self.name = name
        self.model = model
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.other_args = list(other_args or [])
        self.cards = cards or CARD
        self.env_extra = dict(env_extra or {})
        self.timeout_for_server_launch = timeout
        self.log_path = os.path.join(CASES_LOGDIR, f"e2e_{name}.log")
        self.process = None
        self.server_cmd = ""
        self._thread = None
        self._error = None
        self._fh = None

    def start(self):
        from sglang.test.test_utils import popen_launch_server

        os.makedirs(CASES_LOGDIR, exist_ok=True)
        # Mixin env style: {**os.environ, ...NPU-related...}, then overlay
        # case-specific variables.
        # Note: do not set PYTORCH_NPU_ALLOC_CONF=expandable_segments:True — on
        # this machine (torch_npu) that config makes the scheduler crash during
        # the memory pool initialization phase.
        env = {**os.environ, "ASCEND_RT_VISIBLE_DEVICES": self.cards}
        # In source-tree mode the Rust extension of --grpc-port is compiled /
        # cache-validated by the loader via cargo; cargo lives in ~/.cargo/bin
        # (not on a non-interactive shell's PATH).
        cargo_bin = os.path.expanduser("~/.cargo/bin")
        if os.path.isdir(cargo_bin):
            env["PATH"] = cargo_bin + os.pathsep + env.get("PATH", "")
        env.update(self.env_extra)

        self._fh = open(self.log_path, "w", encoding="utf-8")  # noqa: SIM115
        self._fh.write(f"[BASE_URL] {self.base_url}\n")
        self._fh.flush()

        def _launch():
            try:
                self.process = popen_launch_server(
                    self.model,
                    self.base_url,
                    timeout=self.timeout_for_server_launch,
                    other_args=self.other_args,
                    env=env,
                    # Server stdout/stderr tee: write to disk (log file) +
                    # echo to the main process
                    return_stdout_stderr=(self._fh, self._fh),
                )
                self.server_cmd = subprocess.list2cmdline(self.process.args)
            except BaseException as e:  # noqa: BLE001  negative cases depend on this exception
                self._error = e

        self._thread = threading.Thread(target=_launch, daemon=True)
        self._thread.start()
        return self

    # ---- Dedicated launch methods for special timings (special parameters; do not use plain start()) ---- #

    def start_gated(self, gate_port, timeout=None):
        """For --gated-launch-port: wait for the gate port to listen plus the
        blocking marker.

        On return the server launch is still suspended by the gate (the health
        wait of popen_launch_server blocks in the background thread); the
        caller then releases it with activate_gate(gate_port).
        """
        self.start()
        self.wait_port(gate_port, timeout=300)
        self.wait_log("Gated launch waiting for activation.", timeout=timeout or 300)
        return self

    def activate_gate(self, gate_port):
        """POST /gate/activate to release the gate (repeated POSTs are
        idempotent); returns the response for assertions."""
        import requests

        r = requests.post(f"http://127.0.0.1:{gate_port}/gate/activate", timeout=5)
        r.raise_for_status()
        return r

    def start_smg_grpc(self, timeout=None):
        """For --smg-grpc-mode: no /health_generate, so join_launch cannot be
        used.

        Server launch completion is determined by the readiness marker of the
        HTTP sidecar (port+1). The gRPC main port binds later than that
        marker; the caller must wait_port(self.port) before sending RPCs.
        """
        self.start()
        self.wait_log("HTTP sidecar server started on http://", timeout=timeout or 600)
        return self

    def _check_launch_failed(self):
        if self._error is not None:
            raise AssertionError(
                f"{self.name}: server launch failed: {self._error}; log tail:\n"
                f"{self.log()[-3000:]}"
            )

    def join_launch(self, timeout=None):
        """Wait for popen_launch_server to return (/health_generate ready or
        failing as expected)."""
        timeout = timeout or self.timeout_for_server_launch + 120
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError(
                f"{self.name}: server not ready within {timeout}s; log tail:\n{self.log()[-3000:]}"
            )
        self._check_launch_failed()
        return self.process

    def wait_log(self, markers, timeout=None, poll=3):
        """Wait until all markers appear in the log; raise immediately if the
        launch failed."""
        if isinstance(markers, str):
            markers = [markers]
        timeout = timeout or self.timeout_for_server_launch
        deadline = time.time() + timeout
        while time.time() < deadline:
            text = self.log()
            missing = [m for m in markers if m not in text]
            if not missing:
                return text
            if self._error is not None:
                raise AssertionError(
                    f"{self.name}: server launch failed: {self._error}, "
                    f"missing markers={missing}; log tail:\n{text[-3000:]}"
                )
            time.sleep(poll)
        raise TimeoutError(
            f"{self.name}: markers={missing} not seen within {timeout}s; "
            f"log tail:\n{self.log()[-3000:]}"
        )

    def wait_port(self, port, timeout=300, poll=2):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if port_listening(port):
                return True
            self._check_launch_failed()
            time.sleep(poll)
        raise TimeoutError(f"{self.name}: port {port} not listening within {timeout}s")

    def log(self):
        self._flush()
        if not os.path.exists(self.log_path):
            return ""
        with open(self.log_path, encoding="utf-8", errors="replace") as f:
            return f.read()

    def _flush(self):
        if self._fh:
            try:
                self._fh.flush()
            except ValueError:
                pass

    def is_running(self):
        return self._thread is not None and self._thread.is_alive() and not self._error

    def stop(self):
        if self.process is not None:
            from sglang.srt.utils import kill_process_tree

            try:
                kill_process_tree(self.process.pid)
            except Exception:  # noqa: BLE001
                print(
                    f"{self.name}: failed to kill the process tree {self.process.pid}"
                )
        elif self._thread is not None and self._thread.is_alive():
            # popen_launch_server has not returned yet (log-wait mode): the
            # process object lives inside it; fall back to killing the whole
            # child process tree of this test process.
            try:
                import psutil
                from sglang.srt.utils import kill_process_tree

                for child in psutil.Process().children(recursive=True):
                    try:
                        kill_process_tree(child.pid)
                    except Exception:  # noqa: BLE001
                        print(
                            f"{self.name}: failed to kill the child process tree {child.pid}"
                        )
            except Exception:  # noqa: BLE001
                print(
                    f"{self.name}: failed to kill the child process tree of this process"
                )
        self._thread = None
        if self._fh:
            try:
                self._fh.close()
            except ValueError:
                pass
            self._fh = None


class AscendServerTestCase(CustomTestCase):
    """Base class that wires AscendServer into the unittest lifecycle (assign
    cls.server in setUpClass)."""

    server = None

    @classmethod
    def tearDownClass(cls):
        if cls.server:
            cls.server.stop()


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
