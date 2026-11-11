import numpy as np
import pytest

from biyoves.geometry import (
    BIOMETRIC_PRESET,
    SCHENGEN_VISA_PRESET,
    USA_VISA_PRESET,
    VESIKALIK_PRESET,
    _head_region,
    build_placement,
    canvas_size,
    combine_hybrid_masks,
    quality_review_reasons,
    render_composite,
)


def test_photo_presets_use_expected_visible_top_margins():
    assert BIOMETRIC_PRESET.visible_top_margin_mm == 3.0
    assert VESIKALIK_PRESET.visible_top_margin_mm == 3.0
    assert USA_VISA_PRESET.visible_top_margin_mm == 1.0
    assert SCHENGEN_VISA_PRESET.visible_top_margin_mm == 1.0


def test_biometric_canvas_and_quality_gate_thresholds():
    assert canvas_size(600) == (1181, 1417)
    assert canvas_size(600, VESIKALIK_PRESET) == (1063, 1417)
    assert quality_review_reasons({"min_openness": 0.199}) == ["eyes_closed"]
    assert quality_review_reasons({"min_openness": 0.20}) == []
    assert quality_review_reasons(
        {"min_openness": 0.30},
        {"head_right": True},
    ) == ["clipping"]


def test_vesikalik_placement_uses_32mm_head_and_45mm_canvas():
    alpha = np.zeros((80, 100), dtype=np.float32)
    alpha[10:60, 10:35] = 1.0
    geometry = {
        "width": 100,
        "height": 80,
        "eye_mid": np.array([20.0, 20.0]),
        "x_axis": np.array([1.0, 0.0]),
        "y_axis": np.array([0.0, 1.0]),
        "nose_midline": np.array([20.0, 20.0]),
        "chin": np.array([20.0, 90.0]),
        "chin_v": 70.0,
        "roll_degrees": 0.0,
    }

    placement = build_placement(alpha, geometry, 600, "vesikalik")
    diagnostics = placement["diagnostics"]

    assert diagnostics["preset"] == "vesikalik"
    assert diagnostics["canvas_mm"] == {"width": 45.0, "height": 60.0}
    assert diagnostics["canvas_px"] == {"width": 1063, "height": 1417}
    assert diagnostics["target_visible_top_to_chin_mm"] == 32.0


def test_hybrid_mask_has_soft_chin_transition():
    hair = np.zeros((81, 5), dtype=np.float32)
    body = np.ones((81, 5), dtype=np.float32)
    combined, diagnostics = combine_hybrid_masks(
        hair,
        body,
        {"eye_mid": np.array([2.0, 0.0]), "y_axis": np.array([0.0, 1.0]), "chin_v": 40.0},
    )

    assert np.all(combined[0] == 0.0)
    assert np.all(combined[-1] == 1.0)
    assert 0.0 < float(combined[40, 2]) < 1.0
    assert diagnostics["chin_v"] == 40.0


def _single_face_geometry():
    return {
        "width": 100,
        "height": 80,
        "eye_mid": np.array([20.0, 20.0]),
        "x_axis": np.array([1.0, 0.0]),
        "y_axis": np.array([0.0, 1.0]),
        "chin_v": 70.0,
    }


def test_single_face_preserves_soft_disconnected_hair_alpha():
    alpha = np.zeros((80, 100), dtype=np.float32)
    alpha[10:60, 10:35] = 1.0
    alpha[5, 50] = 0.2

    region = _head_region(alpha, _single_face_geometry())

    assert float(region["alpha"][5, 50]) == pytest.approx(0.2)


def test_render_composite_preserves_soft_alpha():
    image = np.full((2, 3, 3), 100, dtype=np.uint8)
    alpha = np.array([[0.0, 0.5, 1.0], [1.0, 0.25, 0.0]], dtype=np.float32)
    placement = {
        "matrix": np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        "region": {"alpha": alpha},
        "diagnostics": {"canvas_px": {"width": 3, "height": 2}},
    }

    output = render_composite(image, placement)

    assert output.shape == image.shape
    assert output.dtype == np.uint8
    assert int(output[0, 0, 0]) == 255
    assert 175 <= int(output[0, 1, 0]) <= 180
    assert int(output[0, 2, 0]) == 100
