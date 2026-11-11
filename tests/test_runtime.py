from types import SimpleNamespace

import numpy as np
import pytest

import biyoves.runtime as runtime
from biyoves.errors import InputProcessingError
from biyoves.geometry import IMAGE_LEFT_EYE, IMAGE_RIGHT_EYE


def test_pipeline_runs_exact_hybrid_resolution_sequence(monkeypatch):
    pipeline = object.__new__(runtime.ProductionPipeline)
    calls = []
    presets = []

    class FakeSegmenter:
        device = "cpu"

        def predict_mask(self, image, input_size):
            calls.append(input_size)
            return np.full(image.shape[:2], 0.8, dtype=np.float32)

    pipeline.segmenter = FakeSegmenter()
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    geometry = {
        "eye_mid": np.array([5.0, 1.0]),
        "y_axis": np.array([0.0, 1.0]),
        "chin_v": 5.0,
    }
    monkeypatch.setattr(
        pipeline,
        "_prepare",
        lambda _image: (image, geometry, {"review_reasons": []}),
    )
    monkeypatch.setattr(runtime, "combine_hybrid_masks", lambda hair, body, _geometry: (hair, {}))
    monkeypatch.setattr(
        runtime,
        "build_placement",
        lambda _mask, _geometry, _dpi, preset: (
            presets.append(preset)
            or {"diagnostics": {"canvas_px": {"width": 1, "height": 1}}}
        ),
    )
    monkeypatch.setattr(runtime, "render_composite", lambda _image, _placement: image)

    outputs, diagnostics = pipeline.process(image, dpi=600, preset="vesikalik")

    # The 992px body mask is identical for all three variants, so it is
    # computed once and reused; only the hair branch runs three times.
    assert calls == [992, 256, 416, 608]
    assert list(outputs) == ["hair_256", "hair_416", "hair_608"]
    assert [
        diagnostics["variants"][key]["hair_input_px"]["width"]
        for key in outputs
    ] == [256, 416, 608]
    assert [preset.name for preset in presets] == ["vesikalik", "vesikalik", "vesikalik"]
    assert diagnostics["preset"] == "vesikalik"


def test_prepare_preserves_single_face_geometry_without_multi_face_fields():
    pipeline = object.__new__(runtime.ProductionPipeline)

    class FakeFaceLandmarker:
        last_elapsed_ms = 1.25

        def detect(self, _image):
            points = np.full((478, 3), (0.5, 0.5, 0.0), dtype=np.float64)
            points[IMAGE_LEFT_EYE, :2] = (0.3, 0.35)
            points[IMAGE_RIGHT_EYE, :2] = (0.7, 0.35)
            points[33, :2] = (0.25, 0.35)
            points[133, :2] = (0.35, 0.35)
            points[160, 1] = 0.33
            points[144, 1] = 0.37
            points[159, 1] = 0.33
            points[145, 1] = 0.37
            points[158, 1] = 0.33
            points[153, 1] = 0.37
            points[263, :2] = (0.65, 0.35)
            points[362, :2] = (0.75, 0.35)
            points[387, 1] = 0.33
            points[373, 1] = 0.37
            points[386, 1] = 0.33
            points[374, 1] = 0.37
            points[385, 1] = 0.33
            points[380, 1] = 0.37
            points[152, :2] = (0.5, 0.8)
            return [points]

    pipeline.face_landmarker = FakeFaceLandmarker()
    pipeline.paths = SimpleNamespace(
        birefnet_snapshot="/tmp/birefnet",
        face_landmarker="/tmp/face_landmarker.task",
    )
    pipeline.segmenter = SimpleNamespace(device="cpu")
    _, geometry, diagnostics = pipeline._prepare(np.zeros((200, 200, 3), dtype=np.uint8))

    assert "face_count" not in geometry
    assert "face_bboxes" not in geometry
    assert "face_count" not in diagnostics
    assert "selected_face_index" not in diagnostics
    assert diagnostics["review_reasons"] == []


def test_prepare_rejects_images_without_a_face():
    pipeline = object.__new__(runtime.ProductionPipeline)

    class EmptyFaceLandmarker:
        last_elapsed_ms = 0.5

        def detect(self, _image):
            return []

    pipeline.face_landmarker = EmptyFaceLandmarker()

    with pytest.raises(InputProcessingError, match="no face detected"):
        pipeline._prepare(np.zeros((200, 200, 3), dtype=np.uint8))
