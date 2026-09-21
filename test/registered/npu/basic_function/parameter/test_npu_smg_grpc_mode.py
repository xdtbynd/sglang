"""--smg-grpc-mode: replace the default HTTP server with the legacy SMG gRPC service.

Verification level: confirmed
Consumption points: launch_server.py:44-51 → entrypoints/grpc_server.py:157 (serve_grpc
delegates to the smg-grpc-servicer package; gRPC binds the --port main port, HTTP sidecar
defaults to port+1)
Server launch style: sglang/test/ascend (popen_launch_server + kill_process_tree).
Note: under smg-grpc-mode the main port is gRPC and the sidecar (port+1) has no
/health_generate, so popen_launch_server's health wait will never become ready — this case
uses AscendServer's dedicated launch method start_smg_grpc() (log marker decides readiness,
no join_launch), and tearDownClass kills the process tree via AscendServer.stop().
Evidence: the gRPC port is really listening and actual RPC calls succeed —
standard grpc.health.v1.Health/Check polled until SERVING (flipped from
NOT_SERVING after warmup completes), SglangScheduler/GetModelInfo returns model info,
SglangScheduler/Generate really generates (tokenized input + streaming response).

[Test Category] Parameter
[Test Target] --smg-grpc-mode
"""

import time
import unittest

import grpc
from para_case_common import (
    COMMON_OTHER_ARGS,
    SMALL,
    AscendServer,
    AscendServerTestCase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=900, suite="nightly-1-npu-a3", nightly=True)


class TestSmgGrpcMode(AscendServerTestCase):
    PORT = 31150
    SIDECAR_PORT = 31151  # smg_http_sidecar_port defaults to --port + 1

    @classmethod
    def setUpClass(cls):
        cls.server = AscendServer(
            "01_smg_grpc_mode",
            SMALL,
            cls.PORT,
            other_args=COMMON_OTHER_ARGS + ["--smg-grpc-mode"],
            timeout=600,
        ).start_smg_grpc()

    def test_smg_grpc_mode_real_server(self):
        """Legacy SMG gRPC server launch: port listening + HealthCheck/GetModelInfo/Generate really usable."""
        from smg_grpc_proto.generated import (
            sglang_scheduler_pb2 as pb,
        )
        from smg_grpc_proto.generated import (
            sglang_scheduler_pb2_grpc as pb_grpc,
        )

        self.assertTrue("gRPC" in self.server.log() or "grpc" in self.server.log())
        self.assertIn("HTTP sidecar server started on http://", self.server.log())
        # sidecar prints before the gRPC bind; port readiness needs polling
        self.server.wait_port(self.PORT, timeout=300)
        self.server.wait_port(self.SIDECAR_PORT, timeout=60)

        channel = grpc.insecure_channel(f"127.0.0.1:{self.PORT}")
        grpc.channel_ready_future(channel).result(timeout=120)
        try:
            # 1) Standard gRPC health (grpc.health.v1): the warmup thread flips the status
            #    to SERVING once the engine is ready (health_servicer.py starts as NOT_SERVING)
            from grpc_health.v1 import health_pb2_grpc

            health_stub = health_pb2_grpc.HealthStub(channel)
            self._wait_serving(health_stub, timeout=300)

            # 2) GetModelInfo: real unary RPC, returns model info
            stub = pb_grpc.SglangSchedulerStub(channel)
            model_info = stub.GetModelInfo(pb.GetModelInfoRequest(), timeout=120)
            self.assertTrue(
                model_info.model_path or model_info.model_name,
                f"GetModelInfo returned empty: {model_info}",
            )

            # 3) Generate: real streaming generation with tokenized input
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(SMALL, trust_remote_code=True)
            prompt = "The capital of France is"
            req = pb.GenerateRequest(
                request_id="para_cases_01",
                tokenized=pb.TokenizedInput(
                    original_text=prompt,
                    input_ids=tok(prompt)["input_ids"],
                ),
                sampling_params=pb.SamplingParams(temperature=0.0, max_new_tokens=8),
            )
            chunks = list(stub.Generate(req, timeout=180))
            self.assertTrue(chunks, "Generate streaming response is empty")
            complete = next(
                (c for c in chunks if c.WhichOneof("response") == "complete"), None
            )
            self.assertIsNotNone(
                complete,
                f"streaming response missing complete frame: {len(chunks)} frames",
            )
            output_ids = list(complete.complete.output_ids)
            self.assertTrue(output_ids, "Generate produced no tokens")
            text = tok.decode(output_ids)
            self.assertTrue(text.strip(), f"generation is empty: ids={output_ids[:8]}")
        finally:
            channel.close()

    @staticmethod
    def _wait_serving(health_stub, timeout):
        """Poll the standard health check from NOT_SERVING to SERVING."""
        from grpc_health.v1 import health_pb2

        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            last = health_stub.Check(
                health_pb2.HealthCheckRequest(service=""), timeout=10
            ).status
            if last == health_pb2.HealthCheckResponse.SERVING:
                return
            time.sleep(3)
        raise AssertionError(
            f"health check did not reach SERVING within {timeout}s: last={last}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
