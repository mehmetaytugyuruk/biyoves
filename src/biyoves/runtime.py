"""Pinned model assets, inference backends, and the processing pipeline."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import shutil
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .errors import InputProcessingError, ModelAssetsError, RuntimeErrorBase
from .geometry import (
    DEFAULT_PRESET_NAME,
    MODEL_INPUT_SIZE,
    PresetLike,
    _eye_openness,
    _face_geometry,
    _select_largest_face,
    build_placement,
    quality_review_reasons,
    render_composite,
)

BIREFNET_REPOSITORY = "ZhengPeng7/BiRefNet_lite-matting"
BIREFNET_REVISION = "99c33412e3f58e1f33187abdc8c435c645243690"
BIREFNET_WEIGHT_SHA256 = "ce8bcfc045e336322c0424a5863dcfb7e9ce8fed0a5fd4d1b2b20adf12d97243"
BIREFNET_FILE_SHA256 = {
    "config.json": "2050e7f1d76417bb167d86a22a52737de2a6e114c0f26c8df85c188366819d72",
    "birefnet.py": "af8568b5be406bf4d2a68a7ed6d72e40f73b37a1fb6fc9ebd71b5b3cbcd069c9",
    "BiRefNet_config.py": "e7b8c2a74f6cea6a59553d517f71d47f2c1d90e670a13416af17c25fe2f3dc52",
    "model.safetensors": BIREFNET_WEIGHT_SHA256,
}

MEDIAPIPE_VERSION = "0.10.35"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_LANDMARKER_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"

@dataclass(frozen=True)
class ModelPaths:
    cache_root: Path
    birefnet_snapshot: Path
    face_landmarker: Path


def cache_root(path: Optional[Path] = None) -> Path:
    """Return the deterministic user cache root used by the application."""
    if path is not None:
        return Path(path).expanduser()
    configured = os.environ.get("BIYOVES_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "biyoves"


def model_paths(path: Optional[Path] = None) -> ModelPaths:
    root = cache_root(path)
    return ModelPaths(
        cache_root=root,
        birefnet_snapshot=root / "models" / "birefnet" / BIREFNET_REVISION,
        face_landmarker=root / "models" / "mediapipe" / "face_landmarker.task",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_file(path: Path, expected: str) -> None:
    if not path.is_file():
        raise ModelAssetsError(f"Missing model asset: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise ModelAssetsError(
            f"Checksum mismatch for {path.name}: expected {expected}, got {actual}"
        )


def validate_birefnet_snapshot(path: Path) -> Path:
    """Validate the pinned BiRefNet files without contacting the network."""
    for filename, expected in BIREFNET_FILE_SHA256.items():
        _validate_file(path / filename, expected)
    return path


def validate_face_landmarker(path: Path) -> Path:
    """Validate the pinned MediaPipe task without contacting the network."""
    _validate_file(path, FACE_LANDMARKER_SHA256)
    return path


def validate_local_assets(paths: ModelPaths) -> ModelPaths:
    """Validate all pinned files without contacting any remote service."""
    try:
        validate_birefnet_snapshot(paths.birefnet_snapshot)
        validate_face_landmarker(paths.face_landmarker)
    except ModelAssetsError as exc:
        raise ModelAssetsError(
            f"{exc}. Start BiyoVes while online once to download the required models."
        ) from exc
    return paths


def _download_url(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as output:
            with urllib.request.urlopen(url, timeout=120) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(chunk)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _download_birefnet(paths: ModelPaths) -> None:
    from huggingface_hub import snapshot_download

    paths.birefnet_snapshot.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{BIREFNET_REVISION}.", dir=str(paths.birefnet_snapshot.parent)))
    try:
        snapshot_download(
            repo_id=BIREFNET_REPOSITORY,
            revision=BIREFNET_REVISION,
            local_dir=str(temporary),
            cache_dir=str(paths.cache_root / "huggingface"),
            allow_patterns=list(BIREFNET_FILE_SHA256),
            local_files_only=False,
        )
        for filename, expected in BIREFNET_FILE_SHA256.items():
            _validate_file(temporary / filename, expected)
        if paths.birefnet_snapshot.exists():
            shutil.rmtree(paths.birefnet_snapshot)
        os.replace(temporary, paths.birefnet_snapshot)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def download_models(path: Optional[Path] = None) -> ModelPaths:
    """Download and verify the pinned BiRefNet and MediaPipe assets."""
    paths = model_paths(path)
    paths.cache_root.mkdir(parents=True, exist_ok=True)

    birefnet_valid = True
    try:
        validate_birefnet_snapshot(paths.birefnet_snapshot)
    except ModelAssetsError:
        birefnet_valid = False
    if not birefnet_valid:
        _download_birefnet(paths)

    try:
        validate_face_landmarker(paths.face_landmarker)
    except ModelAssetsError:
        _download_url(FACE_LANDMARKER_URL, paths.face_landmarker)

    return validate_local_assets(paths)


class MediaPipeFaceLandmarker:
    def __init__(self, paths: ModelPaths) -> None:
        version = importlib.metadata.version("mediapipe")
        if version != MEDIAPIPE_VERSION:
            raise RuntimeErrorBase(
                f"MediaPipe {MEDIAPIPE_VERSION} is required, found {version}"
            )
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except Exception as exc:
            raise RuntimeErrorBase(f"could not load MediaPipe: {exc}") from exc
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(paths.face_landmarker)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=10,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        try:
            self._mp = mp
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
        except Exception as exc:
            raise RuntimeErrorBase(f"could not initialize MediaPipe Face Landmarker: {exc}") from exc
        self.last_elapsed_ms = 0.0

    def detect(self, image: np.ndarray) -> List[np.ndarray]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )
        start = time.perf_counter()
        result = self._landmarker.detect(mp_image)
        self.last_elapsed_ms = (time.perf_counter() - start) * 1000.0
        return [
            np.asarray([[float(point.x), float(point.y), float(point.z)] for point in face], dtype=np.float64)
            for face in result.face_landmarks
        ]

    def close(self) -> None:
        close = getattr(self._landmarker, "close", None)
        if close is not None:
            close()


class BiRefNetMatting:
    def __init__(self, paths: ModelPaths) -> None:
        try:
            import torch
            from torchvision import transforms
        except Exception as exc:
            raise RuntimeErrorBase(f"could not load BiRefNet dependencies: {exc}") from exc

        self._torch = torch
        self._transforms = transforms
        self.device = self._select_device(torch)
        self.dtype = torch.float16 if self.device.type in {"mps", "cuda"} else torch.float32
        module_cache = paths.cache_root / "huggingface" / "modules"
        module_cache.mkdir(parents=True, exist_ok=True)
        previous_module_cache = os.environ.get("HF_MODULES_CACHE")
        previous_hub_offline = os.environ.get("HF_HUB_OFFLINE")
        previous_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")
        os.environ["HF_MODULES_CACHE"] = str(module_cache)
        try:
            from transformers import AutoModelForImageSegmentation

            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            self.model = AutoModelForImageSegmentation.from_pretrained(
                str(paths.birefnet_snapshot),
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception as exc:
            raise RuntimeErrorBase(
                "could not load the pinned BiRefNet snapshot locally: "
                f"{exc}"
            ) from exc
        finally:
            if previous_module_cache is None:
                os.environ.pop("HF_MODULES_CACHE", None)
            else:
                os.environ["HF_MODULES_CACHE"] = previous_module_cache
            if previous_hub_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = previous_hub_offline
            if previous_transformers_offline is None:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
            else:
                os.environ["TRANSFORMERS_OFFLINE"] = previous_transformers_offline
        self.model.to(self.device)
        if self.dtype == torch.float16:
            self.model.half()
        self.model.eval()
        self._transform = transforms.Compose([
            transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def _select_device(torch: Any) -> Any:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def predict_matte(self, image: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).convert("RGB")
        tensor = self._transform(pil).unsqueeze(0).to(self.device, dtype=self.dtype)
        with self._torch.inference_mode():
            predictions = self.model(tensor)
            prediction = predictions[-1] if isinstance(predictions, (list, tuple)) else predictions
            prediction = prediction.sigmoid().cpu()
        matte = self._transforms.ToPILImage()(prediction[0]).resize(
            pil.size,
            getattr(Image, "Resampling", Image).BILINEAR,
        )
        return np.asarray(matte, dtype=np.float32) / 255.0


class ProductionPipeline:
    """One local-only model pair shared by all files in a batch."""

    def __init__(self, path: Optional[Path] = None) -> None:
        paths = validate_local_assets(model_paths(path))
        self.paths = paths
        self.face_landmarker = MediaPipeFaceLandmarker(paths)
        try:
            self.matting = BiRefNetMatting(paths)
        except Exception:
            self.face_landmarker.close()
            raise

    def process(
        self,
        image: np.ndarray,
        dpi: int,
        preset: PresetLike = DEFAULT_PRESET_NAME,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if image.ndim != 3 or image.shape[2] != 3:
            raise InputProcessingError("input image must be a three-channel image")
        height, width = image.shape[:2]
        if min(width, height) < 64:
            raise InputProcessingError(f"image is too small ({width}x{height}px)")
        faces = self.face_landmarker.detect(image)
        selected_index, normalized_points = _select_largest_face(faces)
        points = normalized_points.copy()
        points[:, 0] *= width - 1
        points[:, 1] *= height - 1
        face_bboxes = []
        for face in faces:
            denormalized = face[:, :2].copy()
            denormalized[:, 0] *= width - 1
            denormalized[:, 1] *= height - 1
            x_min, y_min = np.floor(np.min(denormalized, axis=0)).astype(int)
            x_max, y_max = np.ceil(np.max(denormalized, axis=0)).astype(int)
            face_bboxes.append((
                max(0, int(x_min)),
                max(0, int(y_min)),
                min(width - 1, int(x_max)),
                min(height - 1, int(y_max)),
            ))
        eyes = _eye_openness(points)
        geometry = _face_geometry(points, width, height)
        geometry.update({
            "face_index": selected_index,
            "face_count": len(faces),
            "face_bboxes": face_bboxes,
        })
        alpha = self.matting.predict_matte(image)
        placement = build_placement(alpha, geometry, dpi, preset)
        output = render_composite(image, placement)
        diagnostics = placement["diagnostics"]
        face_count = len(faces)
        review_reasons = quality_review_reasons(
            eyes,
            face_count,
            clipping=diagnostics.get("clipping"),
        )
        diagnostics.update({
            "eyes": eyes,
            "input_px": {"width": width, "height": height},
            "face_count": face_count,
            "selected_face_index": selected_index,
            "review_reasons": review_reasons,
            "face_landmarker_inference_ms": round(self.face_landmarker.last_elapsed_ms, 3),
            "model": {
                "repository": BIREFNET_REPOSITORY,
                "revision": BIREFNET_REVISION,
                "weight_sha256": BIREFNET_WEIGHT_SHA256,
                "cache_snapshot": str(self.paths.birefnet_snapshot),
            },
            "mediapipe": {
                "version": MEDIAPIPE_VERSION,
                "task_sha256": FACE_LANDMARKER_SHA256,
                "cache_path": str(self.paths.face_landmarker),
            },
            "inference": {
                "network": False,
                "device": str(self.matting.device),
            },
            "placement_policy": (
                "visible BiRefNet foreground top to MediaPipe landmark 152 chin; "
                "eye-line roll correction; facial-midline horizontal reference"
            ),
        })
        return output, diagnostics

    def close(self) -> None:
        self.face_landmarker.close()

    def __enter__(self) -> "ProductionPipeline":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
