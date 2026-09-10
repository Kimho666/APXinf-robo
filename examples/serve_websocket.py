#!/usr/bin/env python3
"""L3 -- serve a policy over the OpenPI websocket protocol, and call it.

The transport is the thinnest layer here. It takes anything that duck-types
``Policy`` -- an L2 policy, a robot policy, your own object with ``infer`` and
``metadata`` -- and never looks at a wire key itself. What a client must send is
decided entirely by the policy underneath; the server just publishes
``policy.metadata`` on connect so the client can read the contract instead of
guessing it.

This script serves a real policy and drives it with a real client in one
process, because the interesting failure is on the wire: a client sending keys
the policy does not serve gets an error at ``infer`` time, not at connect time.
The last section provokes exactly that. The server stays on the main thread and
the client is pushed to an executor, which is not an arbitrary choice -- see
:class:`ClientOnAThread`.

Requires the ``apxinf_py`` CUDA binding, a checkpoint, and the serving extra::

    pip install -e ".[serve]"
    python examples/serve_websocket.py --model-dir /path/to/checkpoint

Pass ``--serve`` to run a foreground server for an external client instead.
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib

import _common  # noqa: F401  (installs the source-checkout path shim)
from _common import observation_for_policy

from apxinf_robo import build_robot_policy, websocket_server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-dir", required=True, type=pathlib.Path)
    parser.add_argument("--robot", default="franka_libero")
    parser.add_argument("--precision", choices=("auto", "fp8", "bf16", "int8"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--norm-stats", type=pathlib.Path, default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--serve",
        action="store_true",
        help="run a foreground server for an external client instead of the "
        "in-process round trip",
    )
    return parser.parse_args()


class ClientOnAThread:
    """Run the *client* off-thread, because the policy cannot leave its own.

    The obvious arrangement -- server on a background thread, client in the
    foreground -- does not work, and fails in a way worth seeing once: engine
    handles are ``unsendable``, bound to the thread that created them, so the
    first request reaches ``infer`` on the wrong thread and PyO3 panics::

        PySentencePieceTokenizer is unsendable, but sent to another thread

    So the server's event loop stays on the thread that loaded the policy, and
    the blocking client calls are pushed to an executor instead. A deployment
    never has this problem -- it calls ``serve_forever()`` on the loading thread
    and is done. The inversion is here only so one script can show both ends.
    """

    def __init__(self, policy, work) -> None:
        self._policy, self._work = policy, work

    def run(self) -> None:
        asyncio.run(self._serve_until_done())

    async def _serve_until_done(self) -> None:
        import websockets.asyncio.server as ws_server
        from apxinf.serving.websocket import health_check

        policy_server = websocket_server(self._policy, "127.0.0.1", 0)

        # Port 0 lets the OS pick, so the example never collides with whatever
        # is already listening on this box.
        server = await ws_server.serve(
            policy_server.handler,
            "127.0.0.1",
            0,
            compression=None,
            max_size=None,
            process_request=health_check,
        )
        port = server.sockets[0].getsockname()[1]
        print(f"serving on 127.0.0.1:{port}")
        try:
            # While this awaits, the loop is free to run the handler -- on this
            # thread, where the policy lives.
            await asyncio.get_running_loop().run_in_executor(None, self._work, port)
        finally:
            server.close()
            await server.wait_closed()


def client_half(observation, port: int) -> None:
    """The other end: connect, read the published contract, then break it.

    Runs on a worker thread and touches nothing but the socket. ``observation``
    is built by the caller, on the policy's own thread.
    """
    from openpi_client import websocket_client_policy

    client = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", port)
    try:
        # The client learns the contract rather than assuming it. This is the
        # whole reason metadata is published at connect: a deployed client can
        # assert the keys it is about to send.
        metadata = client.get_server_metadata()
        print("\nthe server published, before any inference:")
        for field in ("robot", "image_keys", "state_key", "state_dim", "prompt_key"):
            if field in metadata:
                print(f"  {field:12s} {metadata[field]!r}")

        result = client.infer(observation)
        actions = result["actions"]
        print(f"\nclient.infer -> actions {actions.shape} {actions.dtype}")

        # And the failure that matters. The handshake succeeded; the keys are
        # wrong. Nothing catches this until an inference is attempted, which is
        # why the contract is published for the client to check first.
        print("\nnow the same server, sent keys it does not serve:")
        try:
            client.infer({"wrong/camera": observation[metadata["image_keys"][0]]})
        except Exception as error:  # noqa: BLE001 - the transport re-raises server-side
            first_line = str(error).strip().splitlines()[0]
            print(f"  refused: {first_line}")
    finally:
        client._ws.close()


def round_trip(policy) -> None:
    """Serve this policy and drive it with a real client, in one process."""
    observation = observation_for_policy(policy)
    ClientOnAThread(policy, lambda port: client_half(observation, port)).run()


def main() -> None:
    args = parse_args()

    options = {"device": args.device, "precision": args.precision}
    if args.norm_stats is not None:
        options["norm_stats"] = args.norm_stats
    policy = build_robot_policy(args.robot, args.model_dir, **options)

    try:
        if args.serve:
            server = websocket_server(policy, args.host, args.port)
            print(f"serving {args.robot} on {args.host}:{args.port} -- ^C to stop")
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                print("\nshutting down")
            return
        round_trip(policy)
    finally:
        policy.close()


if __name__ == "__main__":
    main()
