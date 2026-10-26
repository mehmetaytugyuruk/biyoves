from io import BytesIO

from PIL import Image

import biyoves.server as server


class NoopExecutor:
    def submit(self, _function, _job_id):
        return None


def test_session_cleanup_removes_uploaded_photos_and_outputs(monkeypatch, tmp_path):
    work_root = tmp_path / "biyoves-web-session"
    input_path = work_root / "inputs" / "photo.png"
    output_path = work_root / "results" / "job" / "hair_256.png"
    input_path.parent.mkdir(parents=True)
    output_path.parent.mkdir(parents=True)
    input_path.write_bytes(b"source")
    output_path.write_bytes(b"result")
    monkeypatch.setattr(server, "WORK_ROOT", work_root)

    server._cleanup_work_root()

    assert not work_root.exists()


def test_custom_home_is_served():
    response = server.create_app().test_client().get("/")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "BiyoVes" in body
    assert "Biyometrik" in body
    assert "Vesikalık" in body
    assert "50 × 60 milimetre" in body
    assert "45 × 60 milimetre" in body
    assert "baş 36 mm" not in body
    assert "baş 32 mm" not in body
    assert "/assets/styles.css" in body


def test_job_upload_returns_status_endpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_executor", NoopExecutor())
    monkeypatch.setattr(server, "INPUT_ROOT", tmp_path / "inputs")
    monkeypatch.setattr(server, "RESULT_ROOT", tmp_path / "results")
    server.INPUT_ROOT.mkdir()
    server.RESULT_ROOT.mkdir()

    image = BytesIO()
    Image.new("RGB", (120, 160), (40, 70, 100)).save(image, format="PNG")
    image.seek(0)
    response = server.create_app().test_client().post(
        "/api/jobs",
        data={"photo": (image, "portrait.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 202
    payload = response.get_json()
    assert payload["status"] == "queued"
    assert payload["preset"] == "biometric"
    status = server.create_app().test_client().get(f"/api/jobs/{payload['id']}")
    assert status.status_code == 200
    assert status.get_json()["id"] == payload["id"]


def test_job_upload_accepts_vesikalik_preset(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_executor", NoopExecutor())
    monkeypatch.setattr(server, "INPUT_ROOT", tmp_path / "inputs")
    monkeypatch.setattr(server, "RESULT_ROOT", tmp_path / "results")
    server.INPUT_ROOT.mkdir()
    server.RESULT_ROOT.mkdir()

    image = BytesIO()
    Image.new("RGB", (120, 160), (40, 70, 100)).save(image, format="PNG")
    image.seek(0)
    response = server.create_app().test_client().post(
        "/api/jobs",
        data={"photo": (image, "portrait.png"), "preset": "vesikalik"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 202
    assert response.get_json()["preset"] == "vesikalik"


def test_job_upload_rejects_unknown_preset():
    image = BytesIO()
    Image.new("RGB", (120, 160), (40, 70, 100)).save(image, format="PNG")
    image.seek(0)
    response = server.create_app().test_client().post(
        "/api/jobs",
        data={"photo": (image, "portrait.png"), "preset": "unknown"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "Geçersiz fotoğraf türü" in response.get_json()["error"]


def test_output_download_name_uses_original_stem_and_preset(tmp_path):
    output_path = tmp_path / "hair_256.png"
    Image.new("RGB", (2, 2), (255, 255, 255)).save(output_path)
    job = server.Job(
        job_id="download-name",
        input_path=tmp_path / "ignored.png",
        result_dir=tmp_path,
        source_stem="vesikalik_ornek",
        preset="vesikalik",
        status="completed",
        outputs={"hair_256": output_path},
    )
    server._jobs[job.job_id] = job

    response = server.create_app().test_client().get(
        f"/api/jobs/{job.job_id}/outputs/hair_256?download=1"
    )

    assert response.status_code == 200
    assert "vesikalik_ornek_vesikal%C4%B1k.png" in response.headers["Content-Disposition"]


def test_job_upload_rejects_unknown_extension():
    response = server.create_app().test_client().post(
        "/api/jobs",
        data={"photo": (BytesIO(b"not an image"), "notes.txt")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "JPG" in response.get_json()["error"]
