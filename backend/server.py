"""aiohttp WebSocket backend for FlashWorld VR Dream prototype.

Protocol overview:
Server → Client:
    - bootstrap: {"type": "bootstrap", "generationId": "...", "splatUrl": "...", "seedImageUrl": "...", "poseIntervalSeconds": 2.0, "windowSize": 24}
    - generation: {"type": "generation", "generationId": "...", "splatUrl": "...", "latencyMs": 1234, "windowSize": 24}
    - error: {"type": "error", "message": "..."}

Client → Server:
    - poseBatch: {"type": "poseBatch", "poses": [...], "sinceGenerationId": "..."}
      Each pose: {"position": [x, y, z], "quaternion": [qx, qy, qz, qw], "timestamp": 1700000000.123}
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import logging
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from aiohttp import WSMsgType, web

from backend.flashworld_runner import CameraPose, FlashWorldRunner, GenerationResult

logger = logging.getLogger(__name__)


DEFAULT_FX = 450.0
DEFAULT_FY = 450.0
DEFAULT_CX = 256.0
DEFAULT_CY = 256.0


@dataclass
class SessionState:
    runner: FlashWorldRunner
    window_size: int
    current_cameras: List[CameraPose]
    last_generation: GenerationResult
    generation_counter: int
    pose_interval_seconds: float
    pending_poses: List[CameraPose] = field(default_factory=list)
    generation_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VR Dream WebSocket backend.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--pose-interval-seconds", type=float, default=2.0)
    parser.add_argument("--initial-prompt", type=str, default=None)
    parser.add_argument("--initial-image", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--offload-t5", action="store_true")
    parser.add_argument("--offload-vae", action="store_true")
    parser.add_argument("--offload-transformer-during-vae", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--ckpt", type=str, default=None)
    return parser


async def prepare_session_state(
    runner: FlashWorldRunner,
    pose_interval_seconds: float,
    window_size: int,
) -> SessionState:
    loop = asyncio.get_running_loop()
    logger.info("Running initial FlashWorld generation.")
    result = await loop.run_in_executor(None, runner.run_initial_generation)
    logger.info("Initial generation ready: %s", result.generation_id)
    state = SessionState(
        runner=runner,
        window_size=window_size,
        current_cameras=list(result.camera_window),
        last_generation=result,
        generation_counter=1,
        pose_interval_seconds=pose_interval_seconds,
    )
    return state


def create_app(state: SessionState) -> web.Application:
    app = web.Application()
    app["session_state"] = state
    app.router.add_get("/ws", websocket_handler)
    app.router.add_get("/healthz", health_handler)
    app.router.add_get("/", index_handler)

    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    if frontend_dir.exists():
        app.router.add_static("/frontend/", frontend_dir, show_index=False)

    output_root = state.runner.output_root
    app.router.add_static("/assets/", output_root, show_index=True)
    return app


async def health_handler(_: web.Request) -> web.Response:
    return web.Response(text="ok", content_type="text/plain")


async def index_handler(_: web.Request) -> web.Response:
    root_dir = Path(__file__).resolve().parent.parent
    index_path = root_dir / "frontend" / "index.html"
    return web.FileResponse(index_path)


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    state: SessionState = request.app["session_state"]

    client = request.remote or "unknown"
    logger.info("Client %s connected.", client)

    try:
        await send_bootstrap(ws, state)
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                await handle_text_message(ws, state, msg.data)
            elif msg.type == WSMsgType.ERROR:
                logger.error("WebSocket error: %s", ws.exception())
                break
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Unhandled error in WebSocket handler.")
        await send_error(ws, "Internal server error.")
    finally:
        logger.info("Client %s disconnected.", client)

    return ws


async def send_bootstrap(ws: web.WebSocketResponse, state: SessionState) -> None:
    result = state.last_generation
    splat_url = asset_url(state.runner.output_root, result.spz_path)
    seed_image = result.spz_path.parent / "seed_image.png"
    seed_url = asset_url(state.runner.output_root, seed_image)
    payload = {
        "type": "bootstrap",
        "generationId": result.generation_id,
        "splatUrl": splat_url,
        "seedImageUrl": seed_url,
        "poseIntervalSeconds": state.pose_interval_seconds,
        "windowSize": state.window_size,
    }
    await ws.send_json(payload)


async def handle_text_message(
    ws: web.WebSocketResponse,
    state: SessionState,
    data: str,
) -> None:
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        logger.warning("Failed to decode message: %s", data)
        await send_error(ws, "Invalid JSON payload.")
        return

    msg_type = payload.get("type")
    if msg_type == "poseBatch":
        await handle_pose_batch(ws, state, payload)
    else:
        logger.warning("Unsupported message type: %s", msg_type)
        await send_error(ws, f"Unsupported message type: {msg_type}")


async def handle_pose_batch(
    ws: web.WebSocketResponse,
    state: SessionState,
    payload: Mapping[str, Any],
) -> None:
    poses_data = payload.get("poses")
    if not isinstance(poses_data, Sequence):
        await send_error(ws, "poseBatch poses field must be a list.")
        return

    new_poses = []
    for idx, pose_dict in enumerate(poses_data):
        if not isinstance(pose_dict, Mapping):
            logger.debug("Skipping pose %s: not a mapping (%s).", idx, pose_dict)
            continue
        try:
            new_poses.append(pose_from_payload(pose_dict))
        except (KeyError, TypeError, ValueError) as exc:
            logger.debug("Skipping invalid pose %s: %s", idx, exc)

    if not new_poses:
        logger.info("Received poseBatch with 0 valid poses.")
        return

    async with state.pending_lock:
        state.pending_poses.extend(new_poses)
        pending_count = len(state.pending_poses)

    logger.info("Queued %s new poses (pending=%s).", len(new_poses), pending_count)

    if state.generation_lock.locked():
        logger.info("Generation in progress; poses will be used in the next cycle.")
        return

    await trigger_generation(ws, state)


async def trigger_generation(ws: web.WebSocketResponse, state: SessionState) -> None:
    async with state.pending_lock:
        has_pending = bool(state.pending_poses)
    if not has_pending:
        logger.debug("No pending poses; skipping generation trigger.")
        return

    async with state.generation_lock:
        async with state.pending_lock:
            if not state.pending_poses:
                return
            next_window, consumed = build_next_window(state)

        seed_image = seed_image_path(state.last_generation)
        generation_id = f"gen_{state.generation_counter:03d}"
        state.generation_counter += 1

        logger.info(
            "Starting generation %s with %s poses (window=%s).",
            generation_id,
            consumed,
            state.window_size,
        )

        loop = asyncio.get_running_loop()
        start_time = time.perf_counter()

        func = functools.partial(
            state.runner.generate_with_window,
            next_window,
            seed_image,
            generation_id,
        )

        try:
            result = await loop.run_in_executor(None, func)
        except Exception as exc:
            logger.exception("Generation %s failed.", generation_id)
            await send_error(ws, f"Generation failed: {exc}")
            return

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info(
            "Completed generation %s in %sms (frames=%s).",
            generation_id,
            latency_ms,
            len(result.frames),
        )

        state.current_cameras = list(next_window)
        state.last_generation = result

        await ws.send_json(
            {
                "type": "generation",
                "generationId": result.generation_id,
                "splatUrl": asset_url(state.runner.output_root, result.spz_path),
                "latencyMs": latency_ms,
                "windowSize": state.window_size,
            }
        )

    async with state.pending_lock:
        has_more = bool(state.pending_poses)

    if has_more and not state.generation_lock.locked():
        logger.info("Processing queued poses collected during generation.")
        await trigger_generation(ws, state)


def build_next_window(state: SessionState) -> Tuple[List[CameraPose], int]:
    window: List[CameraPose] = []

    if state.current_cameras:
        window.append(state.current_cameras[-1])

    needed = state.window_size - len(window)
    if needed <= 0:
        return window[: state.window_size], 0

    consumed = min(len(state.pending_poses), needed)
    recent = list(state.pending_poses[-consumed:]) if consumed else []

    if consumed:
        del state.pending_poses[-consumed:]

    window.extend(recent)

    if len(window) < state.window_size:
        missing = state.window_size - len(window)
        logger.warning(
            "Not enough poses to fill window (needed %s, padding with last pose).",
            missing,
        )
        pad_source = recent[-1] if recent else (window[-1] if window else state.current_cameras[-1])
        for _ in range(missing):
            window.append(pad_source)

    if len(window) > state.window_size:
        window = window[: state.window_size]

    return window, len(recent)


def pose_from_payload(payload: Mapping[str, Any]) -> CameraPose:
    position = payload.get("position")
    quaternion = payload.get("quaternion")

    if not isinstance(position, Sequence) or len(position) != 3:
        raise ValueError("Pose position must be a sequence of 3 floats.")
    if not isinstance(quaternion, Sequence) or len(quaternion) != 4:
        raise ValueError("Pose quaternion must be a sequence of 4 floats [qx, qy, qz, qw].")

    timestamp_val = payload.get("timestamp")
    timestamp = float(timestamp_val) if timestamp_val is not None else None

    fx = float(payload.get("fx", DEFAULT_FX))
    fy = float(payload.get("fy", DEFAULT_FY))
    cx = float(payload.get("cx", DEFAULT_CX))
    cy = float(payload.get("cy", DEFAULT_CY))

    qx, qy, qz, qw = (float(quaternion[i]) for i in range(4))
    camera_quaternion = (qw, qx, qy, qz)

    return CameraPose(
        position=(float(position[0]), float(position[1]), float(position[2])),
        quaternion=camera_quaternion,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        timestamp=timestamp,
    )


async def send_error(ws: web.WebSocketResponse, message: str) -> None:
    await ws.send_json({"type": "error", "message": message})


def asset_url(root: Path, target: Path) -> str:
    try:
        relative = target.resolve().relative_to(root.resolve())
    except ValueError:
        logger.warning("Asset path %s is not under root %s.", target, root)
        return str(target)
    return f"/assets/{relative.as_posix()}"


def seed_image_path(result: GenerationResult) -> Path:
    candidate = result.spz_path.parent / "seed_image.png"
    return candidate if candidate.exists() else result.image_prompt_path


async def async_main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    runner_kwargs: Dict[str, Any] = {
        "window_size": args.window_size,
        "device": args.device,
        "offload_t5": args.offload_t5,
        "offload_vae": args.offload_vae,
        "offload_transformer_during_vae": args.offload_transformer_during_vae,
    }
    if args.initial_prompt:
        runner_kwargs["initial_prompt"] = args.initial_prompt
    if args.initial_image:
        runner_kwargs["initial_image_path"] = Path(args.initial_image)
    if args.ckpt:
        runner_kwargs["ckpt"] = Path(args.ckpt)

    runner = FlashWorldRunner(**runner_kwargs)
    state = await prepare_session_state(runner, args.pose_interval_seconds, args.window_size)
    app = create_app(state)

    app_runner = web.AppRunner(app)
    await app_runner.setup()
    site = web.TCPSite(app_runner, host=args.host, port=args.port)
    await site.start()

    logger.info("Server running at ws://%s:%s/ws", args.host, args.port)

    stop_event = asyncio.Event()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            # Signals are not supported on Windows event loops.
            pass

    try:
        await stop_event.wait()
    finally:
        await app_runner.cleanup()


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    # Manual test: run `python backend/server.py` then connect with `wscat -c ws://127.0.0.1:8765/ws`.
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
