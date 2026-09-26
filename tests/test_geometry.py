import numpy as np
import pytest

from biyoves.errors import InputProcessingError
from biyoves.geometry import (
    _head_region,
    canvas_size,
    quality_review_reasons,
)


def test_presets_and_quality_gate_thresholds():
    assert canvas_size(300, "biometric") == (591, 709)
    assert canvas_size(300, "vesikalik") == (531, 709)

    assert quality_review_reasons({"min_openness": 0.199}, 1) == ["eyes_closed"]
    assert quality_review_reasons({"min_openness": 0.20}, 1) == []
    assert quality_review_reasons({"min_openness": 0.30}, 1) == []
    assert quality_review_reasons({"min_openness": 0.30}, 2) == ["multiple_faces"]
    assert quality_review_reasons(
        {"min_openness": 0.30},
        1,
        {"head_right": True},
    ) == ["clipping"]


def _multi_face_geometry():
    return {
        "width": 100,
        "height": 80,
        "eye_mid": np.array([20.0, 20.0]),
        "x_axis": np.array([1.0, 0.0]),
        "y_axis": np.array([0.0, 1.0]),
        "chin_v": 70.0,
        "face_count": 2,
        "face_index": 0,
        "face_bboxes": [(10, 10, 30, 40), (60, 10, 80, 40)],
    }


def test_multi_face_matting_selects_component_linked_to_selected_face():
    alpha = np.zeros((80, 100), dtype=np.float32)
    alpha[10:60, 10:35] = 1.0
    alpha[5:75, 60:95] = 1.0

    region = _head_region(alpha, _multi_face_geometry())

    assert region["component_area_px"] == 50 * 25
    assert region["face_component_coverage"] == 1.0


def test_multi_face_matting_rejects_shared_foreground_component():
    alpha = np.zeros((80, 100), dtype=np.float32)
    alpha[10:60, 10:85] = 1.0

    with pytest.raises(InputProcessingError, match="share one foreground component"):
        _head_region(alpha, _multi_face_geometry())
