"""Presets, face geometry, Quality Gate, and image composition."""

from __future__ import annotations

import io
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageOps

from .errors import InputProcessingError

@dataclass(frozen=True)
class Preset:
    name: str
    width_mm: float
    height_mm: float
    preferred_head_height_mm: float


BIOMETRIC_PRESET = Preset(
    name="biometric",
    width_mm=50.0,
    height_mm=60.0,
    preferred_head_height_mm=36.0,
)
VESIKALIK_PRESET = Preset(
    name="vesikalik",
    width_mm=45.0,
    height_mm=60.0,
    preferred_head_height_mm=31.0,
)
PRESETS = {
    BIOMETRIC_PRESET.name: BIOMETRIC_PRESET,
    VESIKALIK_PRESET.name: VESIKALIK_PRESET,
}
DEFAULT_PRESET_NAME = BIOMETRIC_PRESET.name
PresetLike = Union[str, Preset]

VISIBLE_TOP_MM = 3.0
ALPHA_THRESHOLD = 0.5
MODEL_INPUT_SIZE = 1024
CLOSED_THRESHOLD = 0.20
MIN_FACE_COMPONENT_COVERAGE = 0.15

IMAGE_LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
IMAGE_RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
EYE_OPENNESS_GEOMETRY = {
    "image_left": {
        "width": (33, 133),
        "vertical": ((160, 144), (159, 145), (158, 153)),
    },
    "image_right": {
        "width": (263, 362),
        "vertical": ((387, 373), (386, 374), (385, 380)),
    },
}
NOSE_REFERENCE = [1, 4, 6, 168, 195, 197]
CHIN_INDEX = 152


def decode_image(path: Path) -> np.ndarray:
    """Decode an image as BGR after applying its EXIF orientation."""
    try:
        with Image.open(path) as image:
            oriented = ImageOps.exif_transpose(image)
            rgb = oriented.convert("RGB")
            array = np.asarray(rgb)
    except Exception as exc:
        raise InputProcessingError(f"could not decode image: {exc}") from exc
    if array.ndim != 3 or array.shape[2] != 3:
        raise InputProcessingError("decoded image is not an RGB image")
    return cv2.cvtColor(np.ascontiguousarray(array), cv2.COLOR_RGB2BGR)


def resolve_preset(preset: PresetLike = DEFAULT_PRESET_NAME) -> Preset:
    if isinstance(preset, Preset):
        return preset
    try:
        return PRESETS[preset]
    except KeyError as exc:
        raise ValueError(f"unknown preset: {preset}") from exc


def canvas_size(dpi: int, preset: PresetLike = DEFAULT_PRESET_NAME) -> Tuple[int, int]:
    if dpi <= 0:
        raise ValueError("DPI must be a positive integer")
    config = resolve_preset(preset)
    pixels_per_mm = float(dpi) / 25.4
    return (
        int(round(config.width_mm * pixels_per_mm)),
        int(round(config.height_mm * pixels_per_mm)),
    )


def _unit(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1e-9:
        raise InputProcessingError("face geometry contains a zero-length axis")
    return vector / length


def _mean(points: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    return points[np.asarray(indices, dtype=np.int32), :2].mean(axis=0)


def _eye_openness(points: np.ndarray) -> Dict[str, float]:
    """Measure normalized eye openness from denormalized face landmarks."""
    required_index = max(
        index
        for eye in EYE_OPENNESS_GEOMETRY.values()
        for pair in (eye["width"], *eye["vertical"])
        for index in pair
    )
    if points.ndim != 2 or points.shape[1] < 2 or points.shape[0] <= required_index:
        raise InputProcessingError("face landmark set is incomplete")

    scores = {}
    for eye_name, geometry in EYE_OPENNESS_GEOMETRY.items():
        width = float(np.linalg.norm(
            points[geometry["width"][0], :2] - points[geometry["width"][1], :2]
        ))
        if width <= 1e-9:
            raise InputProcessingError("face geometry contains a zero-length axis")
        vertical_distances = [
            float(np.linalg.norm(points[first, :2] - points[second, :2]))
            for first, second in geometry["vertical"]
        ]
        scores[eye_name] = float(np.median(vertical_distances) / width)

    return {
        "image_left_openness": scores["image_left"],
        "image_right_openness": scores["image_right"],
        "min_openness": min(scores.values()),
    }


def quality_review_reasons(
    eyes: Dict[str, float],
    face_count: int,
    clipping: Dict[str, bool] | None = None,
) -> List[str]:
    """Return manual-review reasons for closed eyes, multiple faces, and clipping."""
    min_openness = eyes["min_openness"]
    reasons = []
    if min_openness < CLOSED_THRESHOLD:
        reasons.append("eyes_closed")
    if face_count > 1:
        reasons.append("multiple_faces")
    if clipping and any(clipping.values()):
        reasons.append("clipping")
    return reasons


def _face_geometry(points: np.ndarray, width: int, height: int) -> Dict[str, Any]:
    if points.shape[0] <= CHIN_INDEX:
        raise InputProcessingError("face landmark set is incomplete")
    left_eye = _mean(points, IMAGE_LEFT_EYE)
    right_eye = _mean(points, IMAGE_RIGHT_EYE)
    eye_mid = (left_eye + right_eye) / 2.0
    x_axis = _unit(right_eye - left_eye)
    y_axis = np.array([-x_axis[1], x_axis[0]], dtype=np.float64)
    chin = points[CHIN_INDEX, :2].astype(np.float64)
    facial_axis = _unit(chin - eye_mid)
    nose = _mean(points, NOSE_REFERENCE)
    nose_midline = eye_mid + float(np.dot(nose - eye_mid, facial_axis)) * facial_axis
    return {
        "width": width,
        "height": height,
        "eye_mid": eye_mid,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "chin": chin,
        "chin_v": float(np.dot(chin - eye_mid, y_axis)),
        "nose_midline": nose_midline,
        "roll_degrees": math.degrees(math.atan2(
            float(right_eye[1] - left_eye[1]), float(right_eye[0] - left_eye[0])
        )),
    }


def _select_largest_face(face_points: Sequence[np.ndarray]) -> Tuple[int, np.ndarray]:
    if not face_points:
        raise InputProcessingError("no face detected")
    areas = []
    for points in face_points:
        x_min, y_min = np.min(points[:, :2], axis=0)
        x_max, y_max = np.max(points[:, :2], axis=0)
        areas.append(max(0.0, float(x_max - x_min)) * max(0.0, float(y_max - y_min)))
    index = max(range(len(areas)), key=lambda item: (areas[item], -item))
    return index, face_points[index]


def _component_face_overlap(
    component: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> Tuple[int, float]:
    x_min, y_min, x_max, y_max = bbox
    if x_min > x_max or y_min > y_max:
        return 0, 0.0
    face_area = (x_max - x_min + 1) * (y_max - y_min + 1)
    overlap = int(component[y_min:y_max + 1, x_min:x_max + 1].sum())
    return overlap, overlap / float(face_area)


def _select_foreground_component(
    labels: np.ndarray,
    stats: np.ndarray,
    geometry: Dict[str, Any],
) -> Tuple[int, float]:
    component_count = len(stats) - 1
    if component_count <= 0:
        raise InputProcessingError("BiRefNet matte contains no foreground component")

    face_count = int(geometry.get("face_count", 1))
    if face_count <= 1:
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return largest_label, 1.0

    selected_face_index = int(geometry.get("face_index", 0))
    face_bboxes = geometry.get("face_bboxes", [])
    if selected_face_index >= len(face_bboxes):
        raise InputProcessingError("could not map the selected face to the foreground matte")

    selected_bbox = face_bboxes[selected_face_index]
    candidates = []
    for label in range(1, component_count + 1):
        component = labels == label
        overlap, coverage = _component_face_overlap(component, selected_bbox)
        candidates.append((coverage, overlap, int(stats[label, cv2.CC_STAT_AREA]), label))
    coverage, _, _, selected_label = max(
        candidates,
        key=lambda candidate: (candidate[0], candidate[1], candidate[2], -candidate[3]),
    )
    if coverage < MIN_FACE_COMPONENT_COVERAGE:
        raise InputProcessingError(
            "multiple faces detected, but the selected face has no separate foreground component"
        )

    selected_component = labels == selected_label
    for index, bbox in enumerate(face_bboxes):
        if index == selected_face_index:
            continue
        _, other_coverage = _component_face_overlap(selected_component, bbox)
        if other_coverage >= MIN_FACE_COMPONENT_COVERAGE:
            raise InputProcessingError(
                "multiple faces share one foreground component; output was not generated"
            )
    return selected_label, coverage


def _head_region(alpha: np.ndarray, geometry: Dict[str, Any]) -> Dict[str, Any]:
    if alpha.ndim != 2:
        raise InputProcessingError("BiRefNet did not return a grayscale matte")
    height, width = geometry["height"], geometry["width"]
    if alpha.shape != (height, width):
        alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LINEAR)
    alpha_uint8 = np.rint(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    binary = (alpha_uint8 >= int(round(ALPHA_THRESHOLD * 255.0))).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        raise InputProcessingError("BiRefNet matte contains no foreground component")
    selected_label, face_component_coverage = _select_foreground_component(
        labels,
        stats,
        geometry,
    )
    component = labels == selected_label
    ys, xs = np.where(component)
    if len(xs) == 0:
        raise InputProcessingError("BiRefNet matte foreground component is empty")

    source_points = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
    delta = source_points - geometry["eye_mid"]
    u = delta @ geometry["x_axis"]
    v = delta @ geometry["y_axis"]
    head_selector = v <= geometry["chin_v"]
    if not np.any(head_selector):
        raise InputProcessingError("BiRefNet foreground has no visible head above chin")
    head_u = u[head_selector]
    head_v = v[head_selector]
    top_v = float(head_v.min())
    return {
        "alpha": alpha_uint8,
        "head_u": head_u,
        "head_v": head_v,
        "top_v": top_v,
        "component_area_px": int(component.sum()),
        "head_area_px": int(head_selector.sum()),
        "component_label": selected_label,
        "face_component_coverage": round(face_component_coverage, 6),
    }


def _mapped_bounds(
    u: np.ndarray,
    v: np.ndarray,
    region: Dict[str, Any],
    scale: float,
    preset: Preset,
) -> Dict[str, float]:
    u_reference = float(np.dot(region["nose_midline"] - region["eye_mid"], region["x_axis"]))
    x = preset.width_mm / 2.0 + scale * (u - u_reference)
    y = VISIBLE_TOP_MM + scale * (v - region["top_v"])
    return {"left": float(x.min()), "right": float(x.max()), "top": float(y.min()), "bottom": float(y.max())}


def build_placement(
    alpha: np.ndarray,
    geometry: Dict[str, Any],
    dpi: int,
    preset: PresetLike = DEFAULT_PRESET_NAME,
) -> Dict[str, Any]:
    """Build the pure source-to-canvas placement and diagnostics.

    For source point ``p``, ``u=dot(p-E, x_axis)`` and
    ``v=dot(p-E, y_axis)``. The output map is centered on the preset canvas
    and starts at the fixed 3.0 mm visible-top margin. The preferred scale
    is reduced only when the visible head would exceed the preset width.
    """
    config = resolve_preset(preset)
    region = _head_region(alpha, geometry)
    visible_height_px = float(geometry["chin_v"] - region["top_v"])
    if visible_height_px <= 0:
        raise InputProcessingError("visible head top is not above chin after roll correction")
    preferred_scale = config.preferred_head_height_mm / visible_height_px
    head_u_reference = float(np.dot(
        geometry["nose_midline"] - geometry["eye_mid"], geometry["x_axis"]
    ))
    preferred_head_bounds = _mapped_bounds(
        region["head_u"],
        region["head_v"],
        {
            "nose_midline": geometry["nose_midline"],
            "eye_mid": geometry["eye_mid"],
            "x_axis": geometry["x_axis"],
            "top_v": region["top_v"],
        },
        preferred_scale,
        config,
    )
    scale = preferred_scale
    width_fit_reduction = False
    if preferred_head_bounds["left"] < 0.0 or preferred_head_bounds["right"] > config.width_mm:
        max_horizontal_offset = float(np.max(np.abs(region["head_u"] - head_u_reference)))
        if max_horizontal_offset > 0.0:
            scale = min(preferred_scale, config.width_mm / (2.0 * max_horizontal_offset))
            width_fit_reduction = scale < preferred_scale

    width_px, height_px = canvas_size(dpi, config)
    pixels_per_mm = float(dpi) / 25.4
    x_constant_mm = config.width_mm / 2.0 - scale * float(np.dot(geometry["nose_midline"], geometry["x_axis"]))
    y_constant_mm = VISIBLE_TOP_MM - scale * (
        float(np.dot(geometry["eye_mid"], geometry["y_axis"])) + region["top_v"]
    )
    matrix = np.array([
        [scale * pixels_per_mm * geometry["x_axis"][0], scale * pixels_per_mm * geometry["x_axis"][1], pixels_per_mm * x_constant_mm],
        [scale * pixels_per_mm * geometry["y_axis"][0], scale * pixels_per_mm * geometry["y_axis"][1], pixels_per_mm * y_constant_mm],
    ], dtype=np.float32)

    head_bounds = _mapped_bounds(region["head_u"], region["head_v"], {
        "nose_midline": geometry["nose_midline"],
        "eye_mid": geometry["eye_mid"],
        "x_axis": geometry["x_axis"],
        "top_v": region["top_v"],
    }, scale, config)
    chin_u = float(np.dot(geometry["chin"] - geometry["eye_mid"], geometry["x_axis"]))
    chin_y_mm = VISIBLE_TOP_MM + scale * (geometry["chin_v"] - region["top_v"])
    chin_x_mm = config.width_mm / 2.0 + scale * (
        chin_u - float(np.dot(geometry["nose_midline"] - geometry["eye_mid"], geometry["x_axis"]))
    )
    final_visible_top_to_chin_mm = scale * visible_height_px
    diagnostics = {
        "preset": config.name,
        "dpi": dpi,
        "canvas_mm": {"width": config.width_mm, "height": config.height_mm},
        "canvas_px": {"width": width_px, "height": height_px},
        "visible_top_to_chin_px": round(visible_height_px, 6),
        "preferred_visible_top_to_chin_mm": config.preferred_head_height_mm,
        "final_visible_top_to_chin_mm": round(final_visible_top_to_chin_mm, 6),
        "scale_mm_per_source_px": round(scale, 9),
        "preferred_scale_mm_per_source_px": round(preferred_scale, 9),
        "final_scale_mm_per_source_px": round(scale, 9),
        "width_fit_reduction": width_fit_reduction,
        "visible_top_to_chin_mm": round(final_visible_top_to_chin_mm, 6),
        "visible_top_y_mm": VISIBLE_TOP_MM,
        "chin": {"x_mm": round(chin_x_mm, 6), "y_mm": round(chin_y_mm, 6)},
        "chin_y_mm": round(chin_y_mm, 6),
        "chin_to_bottom_margin_mm": round(config.height_mm - chin_y_mm, 6),
        "preferred_visible_head_bounds_mm": {
            key: round(value, 6) for key, value in preferred_head_bounds.items()
        },
        "visible_head_bounds_mm": {key: round(value, 6) for key, value in head_bounds.items()},
        "visible_head_margins_mm": {
            "top": round(head_bounds["top"], 6),
            "left": round(head_bounds["left"], 6),
            "right": round(config.width_mm - head_bounds["right"], 6),
        },
        "clipping": {
            "head_top": head_bounds["top"] < 0.0,
            "chin_top": chin_y_mm < 0.0,
            "chin_bottom": chin_y_mm > config.height_mm,
            "head_left": head_bounds["left"] < 0.0,
            "head_right": head_bounds["right"] > config.width_mm,
        },
        "matte": {
            "alpha_threshold": ALPHA_THRESHOLD,
            "component_area_px": region["component_area_px"],
            "head_area_px": region["head_area_px"],
            "component_label": region["component_label"],
            "face_component_coverage": region["face_component_coverage"],
        },
        "roll_degrees": round(float(geometry["roll_degrees"]), 6),
        "affine_source_to_canvas": matrix.round(8).tolist(),
    }
    return {"matrix": matrix, "region": region, "diagnostics": diagnostics}


def render_composite(image: np.ndarray, placement: Dict[str, Any]) -> np.ndarray:
    """Warp the source and matte onto a white canvas."""
    diagnostics = placement["diagnostics"]
    canvas_width = int(diagnostics["canvas_px"]["width"])
    canvas_height = int(diagnostics["canvas_px"]["height"])
    matrix = placement["matrix"]
    region = placement["region"]
    warped = cv2.warpAffine(
        image, matrix, (canvas_width, canvas_height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )
    # The thresholded component is geometry-only. Preserve the soft matte for
    # visual compositing so low-alpha/disconnected hair is not discarded.
    source_alpha = region["alpha"]
    warped_alpha = cv2.warpAffine(
        source_alpha, matrix, (canvas_width, canvas_height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(np.float32)[..., None] / 255.0
    return np.clip(
        warped.astype(np.float32) * warped_alpha + 255.0 * (1.0 - warped_alpha),
        0.0,
        255.0,
    ).astype(np.uint8)


def write_output(path: Path, image: np.ndarray, overwrite: bool = False, dpi: int = 300) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}")
    if dpi <= 0:
        raise ValueError("DPI must be a positive integer")
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    extension = suffix if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"} else ".png"
    format_name = {
        ".png": "PNG",
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".webp": "WEBP",
        ".bmp": "BMP",
        ".tif": "TIFF",
        ".tiff": "TIFF",
    }[extension]
    encoded = io.BytesIO()
    try:
        Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(
            encoded,
            format=format_name,
            dpi=(dpi, dpi),
        )
    except Exception as exc:
        raise InputProcessingError(f"could not encode output as {extension}: {exc}") from exc
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded.getvalue())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
