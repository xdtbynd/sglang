"""--grpc-port: native Rust gRPC service starts alongside HTTP.

Verification level: confirmed
Consumption points: entrypoints/http_server.py:2777 (_start_native_grpc_server_for_runtime
→ load_rust_extension), validation at arg_groups/validation_hook.py:241
Prerequisite: Rust toolchain (rustup 1.92.0 / aarch64); in source-tree mode the loader
compiles rust/sglang-grpc on the fly with cargo and caches it (the cargo path is injected
via para_case_common's env).
Server launch style: sglang/test/ascend (popen_launch_server + kill_process_tree).

[Test Category] Parameter
[Test Target] --grpc-port
"""

import argparse
import unittest

from para_case_common import (
    COMMON_OTHER_ARGS,
    SMALL,
    AscendServer,
    AscendServerTestCase,
    generate,
    port_listening,
    server_info,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1200, suite="nightly-1-npu-a3", nightly=True)


class TestNativeGrpcPort(AscendServerTestCase):
    PORT = 31160
    GRPC_PORT = 31161

    @classmethod
    def setUpClass(cls):
        cls.server = AscendServer(
            "02_native_grpc_port",
            SMALL,
            cls.PORT,
            other_args=COMMON_OTHER_ARGS + ["--grpc-port", str(cls.GRPC_PORT)],
            timeout=1200,
        ).start()
        cls.server.join_launch()

    def test_grpc_port_native_server(self):
        """Argument resolution + native gRPC listening + HealthCheck RPC + HTTP coexistence."""
        from sglang.srt.arg_groups.overrides import resolution_result
        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser(add_help=False)
        ServerArgs.add_cli_args(parser)
        sa = ServerArgs.from_cli_args(
            parser.parse_args(
                ["--model-path", SMALL, "--grpc-port", str(self.GRPC_PORT)]
            )
        )
        sa.resolve_once()
        sa.check_server_args()
        self.assertEqual(resolution_result(sa, "grpc_port"), self.GRPC_PORT)

        log = self.server.log()
        # Python-side lifespan finished the server launch + Rust side really bound
        self.assertIn(f"Native gRPC server started on 127.0.0.1:{self.GRPC_PORT}", log)
        self.assertIn(f"gRPC server listening on 127.0.0.1:{self.GRPC_PORT}", log)
        self.assertTrue(port_listening(self.GRPC_PORT))

        # HTTP and gRPC coexist: the HTTP interface still works in the same process
        info = server_info(self.PORT)
        self.assertEqual(info["grpc_port"], self.GRPC_PORT)
        out = generate(self.PORT, "Native grpc side by side ->", max_new_tokens=8)
        self.assertTrue(out["text"])

        # Real gRPC call (no generated stub; generic unary-unary + hand-written serialization):
        # HealthCheckRequest{} is an empty message, HealthCheckResponse{healthy=true} -> b"\x08\x01"
        import grpc

        channel = grpc.insecure_channel(f"127.0.0.1:{self.GRPC_PORT}")
        grpc.channel_ready_future(channel).result(timeout=60)
        health = channel.unary_unary(
            "/sglang.runtime.v1.SglangService/HealthCheck",
            request_serializer=lambda m: m,
            response_deserializer=lambda b: b,
        )
        resp = health(b"", timeout=60)
        channel.close()
        self.assertTrue(resp, "HealthCheck returned an empty response")


if __name__ == "__main__":
    unittest.main(verbosity=2)
