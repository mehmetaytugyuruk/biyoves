"""Face geometry, output presets, and image composition."""

from __future__ import annotations

import io
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, TypedDict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .errors import InputProcessingError


@dataclass(frozen=True)
class CanvasSpec:
    name: str
    width_mm: float
    height_mm: float
    target_head_height_mm: float
    visible_top_margin_mm: float = 3.0


BIOMETRIC_PRESET = CanvasSpec(
    name="biometric",
    width_mm=50.0,
    height_mm=60.0,
    target_head_height_mm=36.0,
)
VESIKALIK_PRESET = CanvasSpec(
    name="vesikalik",
    width_mm=45.0,
    height_mm=60.0,
    target_head_height_mm=32.0,
)
USA_VISA_PRESET = CanvasSpec(
    name="usa_visa",
    width_mm=50.0,
    height_mm=50.0,
    target_head_height_mm=36.0,
    visible_top_margin_mm=1.0,
)
SCHENGEN_VISA_PRESET = CanvasSpec(
    name="schengen_visa",
    width_mm=35.0,
    height_mm=45.0,
    target_head_height_mm=36.0,
    visible_top_margin_mm=1.0,
)

PRESETS = {
    preset.name: preset
    for preset in (
        BIOMETRIC_PRESET,
        VESIKALIK_PRESET,
        USA_VISA_PRESET,
        SCHENGEN_VISA_PRESET,
    )
}
DEFAULT_PRESET_NAME = BIOMETRIC_PRESET.name
Preset = CanvasSpec
PresetLike = str | CanvasSpec

ALPHA_THRESHOLD = 0.5
CLOSED_THRESHOLD = 0.20
HYBRID_TRANSITION_MIN_PX = 12.0
HYBRID_TRANSITION_FRACTION = 0.08
PRINT_SHEET_WIDTH_MM = 100.0
PRINT_SHEET_HEIGHT_MM = 150.0
PRINT_SHEET_GAP_MM = 2.0
PRINT_FRAME_WIDTH_MM = 0.34
PRINT_FRAME_COLOR = (154, 160, 166)

IMAGE_LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
IMAGE_RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]


class EyeGeometry(TypedDict):
    width: Tuple[int, int]
    vertical: Tuple[Tuple[int, int], ...]


EYE_OPENNESS_GEOMETRY: Dict[str, EyeGeometry] = {
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


def resolve_preset(preset: PresetLike = DEFAULT_PRESET_NAME) -> CanvasSpec:
    if isinstance(preset, CanvasSpec):
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


def compose_print_sheet(
    image: np.ndarray,
    dpi: int,
    preset: PresetLike = DEFAULT_PRESET_NAME,
) -> np.ndarray:
    """Place four full-size copies on a portrait 10×15 cm sheet."""
    if dpi <= 0:
        raise ValueError("DPI must be a positive integer")

    config = resolve_preset(preset)
    photo_width, photo_height = canvas_size(dpi, config)
    if image.shape[:2] != (photo_height, photo_width):
        raise ValueError("Photo dimensions do not match the selected preset")

    pixels_per_mm = dpi / 25.4
    sheet_width = int(round(PRINT_SHEET_WIDTH_MM * pixels_per_mm))
    sheet_height = int(round(PRINT_SHEET_HEIGHT_MM * pixels_per_mm))
    requested_gap = int(round(PRINT_SHEET_GAP_MM * pixels_per_mm))
    column_gap = (
        requested_gap
        if photo_width * 2 + requested_gap <= sheet_width
        else 0
    )
    row_gap = (
        requested_gap
        if photo_height * 2 + requested_gap <= sheet_height
        else 0
    )

    grid_width = photo_width * 2 + column_gap
    grid_height = photo_height * 2 + row_gap
    left = (sheet_width - grid_width) // 2
    top = (sheet_height - grid_height) // 2
    frame_width = max(1, int(round(PRINT_FRAME_WIDTH_MM * pixels_per_mm)))

    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    photo = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(sheet)
    for row in range(2):
        for column in range(2):
            x = left + column * (photo_width + column_gap)
            y = top + row * (photo_height + row_gap)
            sheet.paste(photo, (x, y))
            draw.rectangle(
                (x, y, x + photo_width - 1, y + photo_height - 1),
                outline=PRINT_FRAME_COLOR,
                width=frame_width,
            )

    return cv2.cvtColor(np.asarray(sheet), cv2.COLOR_RGB2BGR)


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
    clipping: Dict[str, bool] | None = None,
) -> List[str]:
    """Return manual-review reasons for closed eyes and clipping."""
    min_openness = eyes["min_openness"]
    reasons = []
    if min_openness < CLOSED_THRESHOLD:
        reasons.append("eyes_closed")
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
    chin: np.ndarray = points[CHIN_INDEX, :2].astype(np.float64)
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


def _select_foreground_component(
    stats: np.ndarray,
) -> int:
    component_count = len(stats) - 1
    if component_count <= 0:
        raise InputProcessingError("BiRefNet mask contains no foreground component")
    return 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def combine_hybrid_masks(
    hair_alpha: np.ndarray,
    body_alpha: np.ndarray,
    geometry: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Blend hair and body masks around the MediaPipe chin transition.

    BiRefNet is run at a lower resolution for hair detail and at 992x992 for
    the face/body silhouette. Hair owns the area above the chin, body owns the
    area below it, and a smooth band avoids a visible seam across the jaw.
    """
    hair = np.asarray(hair_alpha, dtype=np.float32)
    body = np.asarray(body_alpha, dtype=np.float32)
    if hair.ndim != 2 or body.ndim != 2 or hair.shape != body.shape:
        raise InputProcessingError("hybrid masks must be same-size grayscale arrays")
    if len(geometry.get("eye_mid", ())) < 2 or len(geometry.get("y_axis", ())) < 2:
        raise InputProcessingError("face geometry is missing the chin transition axis")

    height, width = hair.shape
    rows: np.ndarray = np.arange(height, dtype=np.float32)[:, None]
    columns: np.ndarray = np.arange(width, dtype=np.float32)[None, :]
    eye_mid = np.asarray(geometry["eye_mid"], dtype=np.float32)
    y_axis = np.asarray(geometry["y_axis"], dtype=np.float32)
    v = (columns - eye_mid[0]) * y_axis[0] + (rows - eye_mid[1]) * y_axis[1]
    chin_v = float(geometry["chin_v"])
    half_width = max(HYBRID_TRANSITION_MIN_PX, abs(chin_v) * HYBRID_TRANSITION_FRACTION)
    start = chin_v - half_width
    end = chin_v + half_width
    transition = np.clip((v - start) / max(end - start, 1e-6), 0.0, 1.0)
    transition = transition * transition * (3.0 - 2.0 * transition)
    combined = hair * (1.0 - transition) + body * transition
    return np.clip(combined, 0.0, 1.0).astype(np.float32), {
        "chin_v": chin_v,
        "transition_half_width_px": round(half_width, 6),
    }


def _head_region(alpha: np.ndarray, geometry: Dict[str, Any]) -> Dict[str, Any]:
    if alpha.ndim != 2:
        raise InputProcessingError("BiRefNet did not return a grayscale mask")
    height, width = geometry["height"], geometry["width"]
    if alpha.shape != (height, width):
        alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LINEAR)
    alpha_uint8 = np.rint(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    binary = (alpha_uint8 >= int(round(ALPHA_THRESHOLD * 255.0))).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        raise InputProcessingError("BiRefNet mask contains no foreground component")
    selected_label = _select_foreground_component(stats)
    component = labels == selected_label
    ys, xs = np.where(component)
    if len(xs) == 0:
        raise InputProcessingError("BiRefNet mask foreground component is empty")

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
    # The binary component is for geometry selection, not for destroying the
    # model's soft hair matte. Preserve the complete float alpha so
    # low-confidence strands and their natural feathering survive.
    render_alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    return {
        "alpha": render_alpha,
        "head_u": head_u,
        "head_v": head_v,
        "top_v": top_v,
        "component_area_px": int(component.sum()),
        "head_area_px": int(head_selector.sum()),
        "component_label": selected_label,
    }


def _mapped_bounds(
    u: np.ndarray,
    v: np.ndarray,
    region: Dict[str, Any],
    scale: float,
    canvas: CanvasSpec,
) -> Dict[str, float]:
    u_reference = float(np.dot(region["nose_midline"] - region["eye_mid"], region["x_axis"]))
    x = canvas.width_mm / 2.0 + scale * (u - u_reference)
    y = canvas.visible_top_margin_mm + scale * (v - region["top_v"])
    return {
        "left": float(x.min()),
        "right": float(x.max()),
        "top": float(y.min()),
        "bottom": float(y.max()),
    }


def build_placement(
    alpha: np.ndarray,
    geometry: Dict[str, Any],
    dpi: int,
    preset: PresetLike = DEFAULT_PRESET_NAME,
) -> Dict[str, Any]:
    """Build the pure source-to-canvas placement and diagnostics.

    For source point ``p``, ``u=dot(p-E, x_axis)`` and
    ``v=dot(p-E, y_axis)``. The output map is centered on the selected canvas
    and starts at the preset's visible-top margin. The preset head height is
    enforced; content wider than the canvas is cropped at its side edges.
    """
    config = resolve_preset(preset)
    region = _head_region(alpha, geometry)
    visible_height_px = float(geometry["chin_v"] - region["top_v"])
    if visible_height_px <= 0:
        raise InputProcessingError("visible head top is not above chin after roll correction")
    target_scale = config.target_head_height_mm / visible_height_px
    target_head_bounds = _mapped_bounds(
        region["head_u"],
        region["head_v"],
        {
            "nose_midline": geometry["nose_midline"],
            "eye_mid": geometry["eye_mid"],
            "x_axis": geometry["x_axis"],
            "top_v": region["top_v"],
        },
        target_scale,
        config,
    )
    scale = target_scale

    width_px, height_px = canvas_size(dpi, config)
    pixels_per_mm = float(dpi) / 25.4
    x_constant_mm = config.width_mm / 2.0 - scale * float(
        np.dot(geometry["nose_midline"], geometry["x_axis"])
    )
    y_constant_mm = config.visible_top_margin_mm - scale * (
        float(np.dot(geometry["eye_mid"], geometry["y_axis"])) + region["top_v"]
    )
    matrix = np.array(
        [
            [
                scale * pixels_per_mm * geometry["x_axis"][0],
                scale * pixels_per_mm * geometry["x_axis"][1],
                pixels_per_mm * x_constant_mm,
            ],
            [
                scale * pixels_per_mm * geometry["y_axis"][0],
                scale * pixels_per_mm * geometry["y_axis"][1],
                pixels_per_mm * y_constant_mm,
            ],
        ],
        dtype=np.float32,
    )

    head_bounds = target_head_bounds
    chin_u = float(np.dot(geometry["chin"] - geometry["eye_mid"], geometry["x_axis"]))
    chin_y_mm = config.visible_top_margin_mm + scale * (
        geometry["chin_v"] - region["top_v"]
    )
    chin_x_mm = config.width_mm / 2.0 + scale * (
        chin_u - float(np.dot(geometry["nose_midline"] - geometry["eye_mid"], geometry["x_axis"]))
    )
    final_visible_top_to_chin_mm = scale * visible_height_px
    diagnostics = {
        "preset": config.name,
        "canvas": config.name,
        "dpi": dpi,
        "canvas_mm": {"width": config.width_mm, "height": config.height_mm},
        "canvas_px": {"width": width_px, "height": height_px},
        "visible_top_to_chin_px": round(visible_height_px, 6),
        "target_visible_top_to_chin_mm": config.target_head_height_mm,
        "final_visible_top_to_chin_mm": round(final_visible_top_to_chin_mm, 6),
        "scale_mm_per_source_px": round(scale, 9),
        "target_scale_mm_per_source_px": round(target_scale, 9),
        "final_scale_mm_per_source_px": round(scale, 9),
        "visible_top_to_chin_mm": round(final_visible_top_to_chin_mm, 6),
        "visible_top_y_mm": config.visible_top_margin_mm,
        "chin": {"x_mm": round(chin_x_mm, 6), "y_mm": round(chin_y_mm, 6)},
        "chin_y_mm": round(chin_y_mm, 6),
        "chin_to_bottom_margin_mm": round(config.height_mm - chin_y_mm, 6),
        "target_visible_head_bounds_mm": {
            key: round(value, 6) for key, value in target_head_bounds.items()
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
        },
        "mask": {
            "alpha_threshold": ALPHA_THRESHOLD,
            "component_area_px": region["component_area_px"],
            "head_area_px": region["head_area_px"],
            "component_label": region["component_label"],
        },
        "roll_degrees": round(float(geometry["roll_degrees"]), 6),
        "affine_source_to_canvas": matrix.round(8).tolist(),
    }
    return {"matrix": matrix, "region": region, "diagnostics": diagnostics}


def render_composite(image: np.ndarray, placement: Dict[str, Any]) -> np.ndarray:
    """Warp the source and soft mask onto a white canvas."""
    diagnostics = placement["diagnostics"]
    canvas_width = int(diagnostics["canvas_px"]["width"])
    canvas_height = int(diagnostics["canvas_px"]["height"])
    matrix = placement["matrix"]
    region = placement["region"]
    warped = cv2.warpAffine(
        image, matrix, (canvas_width, canvas_height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )
    # The thresholded component is geometry-only. Preserve the soft mask for
    # visual compositing so soft hair edges remain natural.
    source_alpha = region["alpha"]
    warped_alpha = cv2.warpAffine(
        source_alpha, matrix, (canvas_width, canvas_height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(np.float32)[..., None]
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
    extension = (
        suffix
        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
        else ".png"
    )
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
