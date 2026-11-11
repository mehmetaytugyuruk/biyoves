"""Standalone local web server for the BiyoVes photo pipeline."""

from __future__ import annotations

import atexit
import errno
import http.client
import logging
import mimetypes
import os
import shutil
import tempfile
import threading
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from flask import Flask, jsonify, request, send_file, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.serving import make_server
from werkzeug.utils import secure_filename

from .errors import InputProcessingError, ModelAssetsError, RuntimeErrorBase
from .geometry import (
    DEFAULT_PRESET_NAME,
    PRESETS,
    compose_print_sheet,
    decode_image,
    resolve_preset,
    write_output,
)
from .runtime import HAIR_INPUT_SIZES, ProductionPipeline, download_models

LOGGER = logging.getLogger(__name__)
OUTPUT_DPI = 600
HOST = "127.0.0.1"
PORT = int(os.environ.get("BIYOVES_PORT", "7860"))
PORT_ATTEMPTS = 100
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VARIANTS = tuple(f"hair_{size}" for size in HAIR_INPUT_SIZES)
PRESET_LABELS = {
    "biometric": "biyometrik",
    "vesikalik": "vesikalık",
    "usa_visa": "USA vizesi",
    "schengen_visa": "Schengen vizesi",
}
WEB_ROOT = Path(__file__).with_name("web")
WORK_ROOT = Path(tempfile.mkdtemp(prefix="biyoves-web-"))
INPUT_ROOT = WORK_ROOT / "inputs"
RESULT_ROOT = WORK_ROOT / "results"
INPUT_ROOT.mkdir(parents=True, exist_ok=True)
RESULT_ROOT.mkdir(parents=True, exist_ok=True)


def _cleanup_work_root() -> None:
    try:
        shutil.rmtree(WORK_ROOT)
    except FileNotFoundError:
        pass
    except OSError:
        LOGGER.exception("Could not remove temporary session data at %s", WORK_ROOT)


atexit.register(_cleanup_work_root)


@dataclass
class Job:
    job_id: str
    input_path: Path
    result_dir: Path
    source_stem: str
    preset: str = DEFAULT_PRESET_NAME
    status: str = "queued"
    progress: int = 0
    message: str = "Sırada."
    outputs: dict[str, Path] = field(default_factory=dict)
    print_outputs: dict[str, Path] = field(default_factory=dict)


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()
_pipeline: ProductionPipeline | None = None
_pipeline_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="biyoves-job")


def _friendly_error(exc: Exception) -> str:
    if isinstance(exc, ModelAssetsError):
        return "Modeller indirilemedi. İnternet bağlantınızı kontrol edip tekrar deneyin."
    if isinstance(exc, InputProcessingError):
        message = str(exc).lower()
        if "no face" in message:
            return "Fotoğrafta yüz bulunamadı. Yüzün net ve kadrajda olduğundan emin olun."
        if "too small" in message:
            return "Fotoğraf çözünürlüğü çok düşük. Daha yüksek çözünürlüklü bir fotoğraf seçin."
        return "Fotoğraf işlenemedi. Farklı bir fotoğraf deneyin."
    if isinstance(exc, RuntimeErrorBase):
        return "İşleme motoru başlatılamadı. Uygulamayı yeniden başlatıp tekrar deneyin."
    LOGGER.exception("Photo processing failed")
    return "Beklenmeyen bir hata oluştu. Uygulamayı yeniden başlatıp tekrar deneyin."


def _set_job(job_id: str, **changes: Any) -> None:
    with _jobs_lock:
        job = _jobs[job_id]
        for key, value in changes.items():
            setattr(job, key, value)


def _get_pipeline(progress: Callable[[float], None]) -> ProductionPipeline:
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            progress(0.02)
            paths = download_models()
            _pipeline = ProductionPipeline(_validated_paths=paths)
    return _pipeline


def _run_job(job_id: str) -> None:
    with _jobs_lock:
        job = _jobs[job_id]

    def progress(value: float, message: str = "", **kwargs: Any) -> None:
        # Reserve the final 25% for writing the output PNGs so progress never
        # moves backwards when the runtime callback reaches 100%.
        normalized = 5 + round(max(0.0, min(1.0, value)) * 70)
        _set_job(job_id, progress=min(95, normalized), message="Fotoğraf işleniyor…")

    try:
        _set_job(job_id, status="processing", progress=2, message="Fotoğraf hazırlanıyor…")
        image = decode_image(job.input_path)
        outputs, diagnostics = _get_pipeline(progress).process(
            image,
            dpi=OUTPUT_DPI,
            progress=progress,
            preset=job.preset,
        )
        job.result_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        print_written: dict[str, Path] = {}
        for index, variant in enumerate(VARIANTS, start=1):
            _set_job(
                job_id,
                progress=75 + round((index / len(VARIANTS)) * 20),
                message="Çıktılar hazırlanıyor…",
            )
            path = job.result_dir / f"{variant}.png"
            write_output(path, outputs[variant], dpi=OUTPUT_DPI)
            written[variant] = path
            print_path = job.result_dir / f"{variant}_10x15.png"
            print_sheet = compose_print_sheet(
                outputs[variant], OUTPUT_DPI, preset=job.preset
            )
            write_output(print_path, print_sheet, dpi=OUTPUT_DPI)
            print_written[variant] = print_path

        message = (
            f"Tamamlandı: {PRESET_LABELS[job.preset]} için fotoğraf ve "
            "10×15 baskı çıktıları hazır."
        )
        if diagnostics.get("review_reasons"):
            message += " Fotoğrafı gözden geçirmeniz önerilir."
        _set_job(
            job_id,
            status="completed",
            progress=100,
            message=message,
            outputs=written,
            print_outputs=print_written,
        )
    except Exception as exc:
        _set_job(job_id, status="failed", progress=0, message=_friendly_error(exc))


def _job_payload(job: Job) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "preset": job.preset,
    }
    if job.status == "completed":
        payload["outputs"] = {
            variant: {
                "preview_url": f"/api/jobs/{job.job_id}/outputs/{variant}",
                "download_url": f"/api/jobs/{job.job_id}/outputs/{variant}?download=1",
                "print_sheet_download_url": (
                    f"/api/jobs/{job.job_id}/outputs/{variant}"
                    "?layout=10x15&download=1"
                ),
            }
            for variant in VARIANTS
        }
    return payload


def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

    @app.get("/")
    def index() -> Any:
        return send_from_directory(WEB_ROOT, "index.html")

    @app.get("/assets/<path:filename>")
    def assets(filename: str) -> Any:
        return send_from_directory(WEB_ROOT, filename)

    @app.post("/api/jobs")
    def create_job() -> Any:
        uploaded = request.files.get("photo")
        if uploaded is None or not uploaded.filename:
            return jsonify({"error": "Bir fotoğraf seçin."}), 400
        extension = Path(uploaded.filename).suffix.lower()
        if extension not in IMAGE_EXTENSIONS:
            return jsonify({"error": "JPG, PNG, WEBP veya TIFF formatı kullanın."}), 400
        preset_name = request.form.get("preset", DEFAULT_PRESET_NAME).strip().lower()
        try:
            preset = resolve_preset(preset_name)
        except ValueError:
            available = ", ".join(PRESETS)
            message = f"Geçersiz fotoğraf türü. Kullanılabilir türler: {available}."
            return jsonify({"error": message}), 400

        job_id = uuid.uuid4().hex
        input_path = INPUT_ROOT / f"{job_id}{extension}"
        result_dir = RESULT_ROOT / job_id
        uploaded.save(input_path)
        job = Job(
            job_id=job_id,
            input_path=input_path,
            result_dir=result_dir,
            source_stem=secure_filename(Path(uploaded.filename).stem) or "biyoves",
            preset=preset.name,
        )
        with _jobs_lock:
            _jobs[job_id] = job
        _executor.submit(_run_job, job_id)
        return jsonify(_job_payload(job)), 202

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id: str) -> Any:
        with _jobs_lock:
            job = _jobs.get(job_id)
        if job is None:
            return jsonify({"error": "İş bulunamadı."}), 404
        return jsonify(_job_payload(job))

    @app.get("/api/jobs/<job_id>/outputs/<variant>")
    def output(job_id: str, variant: str) -> Any:
        with _jobs_lock:
            job = _jobs.get(job_id)
        if job is None:
            return jsonify({"error": "Çıktı bulunamadı."}), 404
        print_sheet = request.args.get("layout") == "10x15"
        path = (job.print_outputs if print_sheet else job.outputs).get(variant)
        if path is None or variant not in VARIANTS or not path.is_file():
            return jsonify({"error": "Çıktı bulunamadı."}), 404
        suffix = "_10x15" if print_sheet else ""
        return send_file(
            path,
            mimetype=mimetypes.types_map.get(".png", "image/png"),
            as_attachment=request.args.get("download") == "1",
            download_name=(
                f"{job.source_stem}_{PRESET_LABELS[job.preset]}{suffix}.png"
            ),
        )

    @app.errorhandler(RequestEntityTooLarge)
    def upload_too_large(_error: RequestEntityTooLarge) -> Any:
        return jsonify({"error": "Fotoğraf boyutu 25 MB sınırını aşamaz."}), 413

    return app


def _is_biyoves_running(port: int) -> bool:
    connection = http.client.HTTPConnection(HOST, port, timeout=1)
    try:
        connection.request("GET", "/")
        response = connection.getresponse()
        return response.status == 200 and b"<title>BiyoVes</title>" in response.read(4096)
    except OSError:
        return False
    finally:
        connection.close()


def _open_browser(url: str) -> None:
    if os.environ.get("BIYOVES_NO_BROWSER") != "1":
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()


def main() -> None:
    app = create_app()
    server = None
    last_port = min(PORT + PORT_ATTEMPTS, 65536)
    try:
        for port in range(PORT, last_port):
            try:
                server = make_server(HOST, port, app, threaded=True)
                # Drain active requests before removing files they may be serving.
                server.daemon_threads = False
                break
            except OSError as exc:
                if exc.errno != errno.EADDRINUSE and getattr(exc, "winerror", None) != 10048:
                    raise
                if _is_biyoves_running(port):
                    url = f"http://{HOST}:{port}"
                    print(f"BiyoVes is already running at {url}", flush=True)
                    _open_browser(url)
                    return

        if server is None:
            raise SystemExit(f"No free port found between {PORT} and {last_port - 1}.")

        url = f"http://{HOST}:{server.server_port}"
        if server.server_port != PORT:
            print(f"Port {PORT} is in use; using {server.server_port} instead.", flush=True)
        print(f"BiyoVes is running at {url}", flush=True)
        _open_browser(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nBiyoVes stopped.", flush=True)
    finally:
        try:
            if server is not None:
                server.server_close()
        finally:
            try:
                _executor.shutdown(wait=True, cancel_futures=True)
            finally:
                try:
                    if _pipeline is not None:
                        _pipeline.close()
                finally:
                    _cleanup_work_root()


if __name__ == "__main__":
    main()
