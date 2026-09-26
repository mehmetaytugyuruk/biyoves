from pathlib import Path
import zipfile

import gradio as gr
import numpy as np
from PIL import Image

import app


class FakePipeline:
    instances = 0
    calls = 0

    def __init__(self):
        type(self).instances += 1

    def process(self, image, dpi, preset):
        type(self).calls += 1
        return np.full((90, 60, 3), 255, dtype=np.uint8), {
            "review_reasons": ["eyes_closed"] if type(self).calls == 3 else [],
        }


def test_single_and_batch_downloads_reuse_pipeline(tmp_path, monkeypatch):
    single = tmp_path / "single.png"
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    for path in (single, first, second):
        Image.new("RGB", (120, 160), (40, 70, 100)).save(path)

    FakePipeline.instances = 0
    FakePipeline.calls = 0
    monkeypatch.setattr(app, "ProductionPipeline", FakePipeline)
    monkeypatch.setattr(app, "_pipeline", None)

    single_result, single_status = app._process_uploads(
        str(single), "biometric", gr.Progress()
    )
    single_output = Path(single_result["value"])
    assert single_output.name == "single.png"
    assert Image.open(single_output).size == (60, 90)
    assert "1/1" in single_status

    batch_result, batch_status = app._process_uploads(
        [str(first), str(second)], "vesikalik", gr.Progress()
    )
    folder_name = Path(batch_result["value"]).stem
    with zipfile.ZipFile(batch_result["value"]) as archive:
        assert f"{folder_name}/first.png" in archive.namelist()
        assert f"{folder_name}/second.png" in archive.namelist()
        assert f"{folder_name}/quality-warnings.txt" in archive.namelist()
    assert "Gözler kapalı veya neredeyse kapalı görünüyor" in batch_status
    assert not single_output.exists()
    assert FakePipeline.instances == 1
    assert FakePipeline.calls == 3


def test_simple_local_interface(monkeypatch):
    demo = app.create_app()
    config = demo.get_config_file()
    components = config["components"]
    upload = next(component for component in components if component["props"].get("label") == "1. Dosya yükleyin")
    download = next(component for component in components if component["props"].get("label") == "3. Çıktıları indir")
    assert upload["props"]["file_count"] == "directory"
    assert download["type"] == "downloadbutton"
    assert not any(component["type"] in {"image", "gallery"} for component in components)
    captured = {}
    monkeypatch.setattr(gr.Blocks, "launch", lambda _demo, **kwargs: captured.update(kwargs))
    app.main()
    assert captured == {"server_name": "127.0.0.1", "share": False}
