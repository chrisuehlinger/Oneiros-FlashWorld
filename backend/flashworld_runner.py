from __future__ import annotations

import argparse
import logging
import math
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from imageio import v2 as imageio_v2

from app import GenerationSystem
from utils import export_gaussians


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraPose:
    position: Tuple[float, float, float]
    quaternion: Tuple[float, float, float, float]
    fx: float
    fy: float
    cx: float
    cy: float
    timestamp: Optional[float] = None

    def to_tensor(self, image_width: int, image_height: int) -> torch.Tensor:
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image_width and image_height must be positive.")

        data = (
            self.quaternion[0],
            self.quaternion[1],
            self.quaternion[2],
            self.quaternion[3],
            self.position[0],
            self.position[1],
            self.position[2],
            self.fx / image_width,
            self.fy / image_height,
            self.cx / image_width,
            self.cy / image_height,
        )
        return torch.tensor(data, dtype=torch.float32)

    def to_json(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "position": list(self.position),
            "quaternion": list(self.quaternion),
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
        }
        if self.timestamp is not None:
            payload["timestamp"] = self.timestamp
        return payload

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "CameraPose":
        return cls(
            position=tuple(float(v) for v in payload["position"]),  # type: ignore[arg-type]
            quaternion=tuple(float(v) for v in payload["quaternion"]),  # type: ignore[arg-type]
            fx=float(payload["fx"]),
            fy=float(payload["fy"]),
            cx=float(payload["cx"]),
            cy=float(payload["cy"]),
            timestamp=float(payload["timestamp"]) if "timestamp" in payload else None,
        )

    def with_intrinsics(self, fx: float, fy: float, cx: float, cy: float) -> "CameraPose":
        return replace(self, fx=fx, fy=fy, cx=cx, cy=cy)


@dataclass
class GenerationResult:
    spz_path: Path
    video_path: Path
    frames: List[Path]
    camera_window: List[CameraPose]
    image_prompt_path: Path
    generation_id: str
    duration_seconds: float


class FlashWorldRunner:
    def __init__(
        self,
        window_size: int = 24,
        output_root: Path | str = Path("output") / "vrdream",
        initial_prompt: str = "A dimly lit bar with nobody inside, neon reflections glimmering on rain-slick wood and drifting dust motes in the air",
        initial_image_path: Path | str = Path("input/seed.png"),
        *,
        ckpt: Optional[Path | str] = None,
        device: str = "cuda:0",
        offload_t5: bool = False,
        offload_vae: bool = False,
        offload_transformer_during_vae: bool = False,
    ) -> None:
        self.window_size = window_size
        self.output_root = Path(output_root).resolve()
        self.initial_prompt = initial_prompt
        # TODO: ensure the seed PNG asset exists prior to running the prototype.
        self.initial_image_path = Path(initial_image_path)
        self.ckpt = Path(ckpt).resolve() if ckpt is not None else None
        self.device = device
        self.offload_t5 = offload_t5
        self.offload_vae = offload_vae
        self.offload_transformer_during_vae = offload_transformer_during_vae

        self._generation_system: Optional[GenerationSystem] = None
        self.output_root.mkdir(parents=True, exist_ok=True)

    def prepare_initial_cameras(self) -> List[CameraPose]:
        radius = 3.0
        height = 1.6
        fov_radians = math.radians(65)

        target_width = 704.0
        target_height = 480.0
        focal_length = 0.5 * target_width / math.tan(fov_radians / 2.0)
        cx = target_width / 2.0
        cy = target_height / 2.0

        target = np.array([0.0, 0.4, 0.0], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        cameras = []
        for idx in range(self.window_size):
            yaw = 2.0 * math.pi * idx / self.window_size
            position = np.array(
                [
                    radius * math.cos(yaw),
                    height,
                    radius * math.sin(yaw),
                ],
                dtype=np.float32,
            )

            quaternion = self._look_at_quaternion(position, target, up)

            cameras.append(
                CameraPose(
                    position=(float(position[0]), float(position[1]), float(position[2])),
                    quaternion=quaternion,
                    fx=float(focal_length),
                    fy=float(focal_length),
                    cx=float(cx),
                    cy=float(cy),
                )
            )

        return cameras

    def run_initial_generation(
        self,
        *,
        generation_id: str = "gen_000",
        resolution: Optional[Tuple[int, int, int]] = None,
    ) -> GenerationResult:
        camera_window = self.prepare_initial_cameras()
        return self._invoke_flashworld(
            camera_window=camera_window,
            text_prompt=self.initial_prompt,
            image_prompt_path=self.initial_image_path,
            generation_id=generation_id,
            resolution=resolution,
        )

    def generate_with_window(
        self,
        camera_window: Sequence[CameraPose],
        seed_image_path: Path,
        generation_id: str,
        *,
        text_prompt: Optional[str] = None,
        resolution: Optional[Tuple[int, int, int]] = None,
    ) -> GenerationResult:
        if len(camera_window) != self.window_size:
            raise ValueError(
                f"Expected camera window of size {self.window_size}, received {len(camera_window)}."
            )

        prompt = text_prompt or self.initial_prompt
        return self._invoke_flashworld(
            camera_window=list(camera_window),
            text_prompt=prompt,
            image_prompt_path=seed_image_path,
            generation_id=generation_id,
            resolution=resolution,
        )

    def _invoke_flashworld(
        self,
        camera_window: Sequence[CameraPose],
        text_prompt: str,
        image_prompt_path: Path | str,
        generation_id: str,
        resolution: Optional[Tuple[int, int, int]],
    ) -> GenerationResult:
        system = self._get_generation_system()
        output_dir = (self.output_root / generation_id).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        image_prompt_path = Path(image_prompt_path).resolve()
        if not image_prompt_path.exists():
            raise FileNotFoundError(f"Prompt image not found at {image_prompt_path}")

        if resolution is None:
            resolution = (self.window_size, 480, 704)
        else:
            if resolution[0] != self.window_size:
                raise ValueError(
                    f"Resolution must include window size {self.window_size} as the first element."
                )

        n_frame, image_height, image_width = resolution
        image_tensor, adjusted_cameras = self._prepare_prompt_image_and_cameras(
            image_prompt_path, camera_window, image_width, image_height
        )

        cameras_tensor = torch.stack(
            [pose.to_tensor(image_width, image_height) for pose in adjusted_cameras],
            dim=0,
        )

        video_path = (output_dir / "video.mp4").resolve()
        spz_path = (output_dir / "gaussians.spz").resolve()

        logger.info("Starting FlashWorld generation %s.", generation_id)
        start_time = time.perf_counter()
        scene_params, ref_w2c, T_norm = system.generate(
            cameras=cameras_tensor,
            n_frame=n_frame,
            image=image_tensor,
            text=text_prompt,
            image_index=0,
            image_height=image_height,
            image_width=image_width,
            video_path=str(video_path),
        )
        duration = time.perf_counter() - start_time
        logger.info("Finished generation %s in %.2f seconds.", generation_id, duration)

        scene_params = scene_params.detach().cpu()

        export_gaussians(
            scene_params,
            opacity_threshold=0.0,
            T_norm=T_norm,
            spz_path=str(spz_path),
            ply_path=None,  # TODO: expose optional PLY export for debugging.
        )

        frames = self._extract_video_frames(video_path, frames_dir)
        seed_frame_path = (output_dir / "seed_image.png").resolve()
        if frames:
            shutil.copy2(frames[0], seed_frame_path)
        else:
            shutil.copy2(image_prompt_path, seed_frame_path)

        generation_result = GenerationResult(
            spz_path=spz_path,
            video_path=video_path,
            frames=frames,
            camera_window=list(adjusted_cameras),
            image_prompt_path=image_prompt_path,
            generation_id=generation_id,
            duration_seconds=duration,
        )

        return generation_result

    def _extract_video_frames(self, video_path: Path, frames_dir: Path) -> List[Path]:
        frame_paths: List[Path] = []
        try:
            reader = imageio_v2.get_reader(video_path)
        except FileNotFoundError:
            logger.warning("Video not found at %s; skipping frame extraction.", video_path)
            return frame_paths

        try:
            for idx, frame in enumerate(reader):
                frame_image = Image.fromarray(frame)
                frame_path = (frames_dir / f"frame_{idx:03d}.png").resolve()
                frame_image.save(frame_path)
                frame_paths.append(frame_path)
        finally:
            reader.close()

        return frame_paths

    def _prepare_prompt_image_and_cameras(
        self,
        image_path: Path,
        camera_window: Sequence[CameraPose],
        image_width: int,
        image_height: int,
    ) -> Tuple[torch.Tensor, List[CameraPose]]:
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size

        scale = max(image_height / original_height, image_width / original_width)
        new_height = int(image_height / scale)
        new_width = int(image_width / scale)

        top = max(0, (original_height - new_height) // 2)
        left = max(0, (original_width - new_width) // 2)

        cropped = image.crop(
            (
                left,
                top,
                left + new_width,
                top + new_height,
            )
        ).resize((image_width, image_height), Image.BICUBIC)

        adjusted_cameras = [
            pose.with_intrinsics(
                fx=pose.fx * scale,
                fy=pose.fy * scale,
                cx=(pose.cx - left) * scale,
                cy=(pose.cy - top) * scale,
            )
            for pose in camera_window
        ]

        image_array = np.array(cropped).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1) * 2 - 1

        return image_tensor.float(), adjusted_cameras

    def _get_generation_system(self) -> GenerationSystem:
        if self._generation_system is None:
            ckpt_path = self._resolve_checkpoint()
            logger.info("Loading GenerationSystem (ckpt=%s).", ckpt_path)
            self._generation_system = GenerationSystem(
                ckpt_path,
                device=self.device,
                offload_t5=self.offload_t5,
                offload_vae=self.offload_vae,
                offload_transformer_during_vae=self.offload_transformer_during_vae,
            )
        return self._generation_system

    def _resolve_checkpoint(self) -> Optional[str]:
        if self.ckpt is not None:
            return str(self.ckpt)

        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
        except ImportError as exc:  # pragma: no cover - optional dependency
            logger.warning(
                "huggingface_hub is not available; proceeding without explicit checkpoint (%s).",
                exc,
            )
            return None

        cache_path = (
            Path(HUGGINGFACE_HUB_CACHE)
            / "models--imlixinyang--FlashWorld"
            / "snapshots"
            / "6a8e88c6f88678ac098e4c82675f0aee555d6e5d"
            / "model.ckpt"
        )

        if not cache_path.exists():
            logger.info("Downloading FlashWorld checkpoint via HuggingFace Hub.")
            hf_hub_download(
                repo_id="imlixinyang/FlashWorld",
                filename="model.ckpt",
                local_dir_use_symlinks=False,
            )

        return str(cache_path)

    @staticmethod
    def _look_at_quaternion(
        position: np.ndarray,
        target: np.ndarray,
        up: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        forward = target - position
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-6:
            forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            forward = forward / forward_norm

        right = np.cross(up, forward)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right = right / right_norm

        true_up = np.cross(forward, right)

        rotation_matrix = np.stack([right, true_up, -forward], axis=1)

        return FlashWorldRunner._matrix_to_quaternion(rotation_matrix)

    @staticmethod
    def _matrix_to_quaternion(matrix: np.ndarray) -> Tuple[float, float, float, float]:
        m = matrix
        trace = np.trace(m)

        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s

        quaternion = np.array([w, x, y, z], dtype=np.float32)
        quaternion /= np.linalg.norm(quaternion)
        return tuple(float(v) for v in quaternion)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FlashWorld backend smoke test.")
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--offload_t5", action="store_true")
    parser.add_argument("--offload_vae", action="store_true")
    parser.add_argument("--offload_transformer_during_vae", action="store_true")
    parser.add_argument("--window_size", type=int, default=24)
    parser.add_argument("--output_root", type=Path, default=Path("output") / "vrdream")
    parser.add_argument("--log_level", type=str, default="INFO")
    return parser.parse_args()


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


if __name__ == "__main__":
    args = _parse_args()
    _configure_logging(args.log_level)

    runner = FlashWorldRunner(
        window_size=args.window_size,
        output_root=args.output_root,
        ckpt=args.ckpt,
        device=args.device,
        offload_t5=args.offload_t5,
        offload_vae=args.offload_vae,
        offload_transformer_during_vae=args.offload_transformer_during_vae,
    )

    try:
        initial_result = runner.run_initial_generation()
        seed_path = initial_result.frames[0] if initial_result.frames else initial_result.image_prompt_path

        runner.generate_with_window(
            initial_result.camera_window,
            seed_image_path=seed_path,
            generation_id="gen_001",
        )
    except Exception as exc:  # pragma: no cover - manual execution path
        logger.exception("FlashWorld runner failed: %s", exc)
