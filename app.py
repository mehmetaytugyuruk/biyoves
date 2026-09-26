"""Simple local Gradio interface for biometric and vesikalık photos."""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

import gradio as gr

from biyoves.geometry import decode_image, write_output
from biyoves.runtime import ModelAssetsError, ProductionPipeline, download_models


_pipeline: ProductionPipeline | None = None
# Keep the current download alive until the process exits or a newer result
# replaces it; TemporaryDirectory removes the root on process shutdown.
_result_workspace = tempfile.TemporaryDirectory(prefix="biyoves-results-")
_result_root = Path(_result_workspace.name)
_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
_QUALITY_WARNINGS = {
    "eyes_closed": "Gözler kapalı veya neredeyse kapalı görünüyor.",
    "multiple_faces": "Birden fazla yüz algılandı; en büyük yüz işlendi.",
    "clipping": "Kafa veya çene kadraj sınırına taşıyor; çıktıyı gözden geçirin.",
}


def _cleanup_result_dirs(keep: Path) -> None:
    for path in _result_root.iterdir():
        if path != keep and path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


def _discard_result_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _get_pipeline(progress: gr.Progress | None = None) -> ProductionPipeline:
    global _pipeline
    if _pipeline is None:
        try:
            _pipeline = ProductionPipeline()
        except ModelAssetsError:
            if progress is not None:
                progress(0, desc="İlk kullanım için modeller indiriliyor…")
            download_models()
            _pipeline = ProductionPipeline()
    return _pipeline


def _uploaded_paths(value) -> list[Path]:
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    paths: list[Path] = []
    for item in values:
        path = Path(str(getattr(item, "name", item)))
        if path.is_dir():
            paths.extend(candidate for candidate in path.rglob("*") if candidate.is_file())
        elif path.is_file():
            paths.append(path)
    return sorted(
        {path for path in paths if path.suffix.lower() in _IMAGE_EXTENSIONS},
        key=lambda path: (path.name.casefold(), str(path).casefold()),
    )


def _process_one(
    source: Path,
    preset: str,
    output: Path,
    progress: gr.Progress,
) -> dict:
    image = decode_image(source)
    processed, diagnostics = _get_pipeline(progress).process(
        image,
        dpi=300,
        preset=preset,
    )
    write_output(output, processed, dpi=300)
    return diagnostics


def _process_uploads(files, preset: str, progress: gr.Progress = gr.Progress()):
    paths = _uploaded_paths(files)
    if not paths:
        return gr.update(value=None, interactive=False), "Önce bir fotoğraf veya klasör yükleyin."

    work_dir = Path(tempfile.mkdtemp(prefix="run-", dir=str(_result_root)))
    total = len(paths)
    successes: list[tuple[Path, Path]] = []
    reports: list[str] = []

    for index, source in enumerate(paths, start=1):
        progress((index - 1) / total, desc=f"{index - 1}/{total} fotoğraf işlendi")
        output = work_dir / f"{index:03d}_{source.name}"
        try:
            diagnostics = _process_one(source, preset, output, progress)
            successes.append((source, output))
            for reason in diagnostics.get("review_reasons", []):
                message = _QUALITY_WARNINGS.get(reason, reason.replace("_", " "))
                reports.append(f"{source.name}: {message}")
        except Exception as exc:
            reports.append(f"{source.name}: İşlenemedi — {exc}")
            if _pipeline is None:
                break

    progress(1, desc=f"{len(successes)}/{total} fotoğraf işlendi")
    if not successes:
        _discard_result_dir(work_dir)
        return gr.update(value=None, interactive=False), "Hiçbir fotoğraf işlenemedi.\n\n" + "\n".join(reports)

    if total == 1 and len(successes) == 1:
        result = work_dir / paths[0].name
        successes[0][1].replace(result)
    else:
        folder_name = "processed"
        result = work_dir / f"{folder_name}.zip"
        with zipfile.ZipFile(result, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            used_names: set[str] = set()
            for index, (source, output) in enumerate(successes, start=1):
                source_name = source.name
                archive_name = f"{folder_name}/{source_name}"
                if source_name in used_names:
                    archive_name = f"{folder_name}/{index:03d}/{source_name}"
                used_names.add(source_name)
                archive.write(output, arcname=archive_name)
            if reports:
                archive.writestr(
                    f"{folder_name}/quality-warnings.txt",
                    "\n".join(reports) + "\n",
                )

    _cleanup_result_dirs(keep=work_dir)

    status = f"Tamamlandı: {len(successes)}/{total} fotoğraf işlendi."
    if reports:
        status += "\n\n" + "\n".join(reports)
    elif successes:
        status += " Kalite uyarısı yok."
    return gr.update(value=str(result), interactive=True), status


def create_app() -> gr.Blocks:
    with gr.Blocks(title="BiyoVes") as demo:
        with gr.Column(elem_id="biyoves-card"):
            gr.Markdown(
                "# BiyoVes\n"
                "Tek fotoğrafı bırakın veya klasör yükleyin. Alt klasörlerdeki "
                "fotoğraflar da işlenir."
            )
            files = gr.File(
                label="1. Dosya yükleyin",
                file_count="directory",
                type="filepath",
                height=150,
            )
            preset = gr.Radio(
                choices=[
                    ("Biometrik · 50 × 60 mm", "biometric"),
                    ("Vesikalık · 45 × 60 mm", "vesikalik"),
                ],
                value="biometric",
                label="Fotoğraf türü",
                container=False,
            )
            start = gr.Button("2. Başla", variant="primary", size="lg")
            status = gr.Markdown("Hazır.")
            download = gr.DownloadButton(
                "3. Çıktıları indir",
                interactive=False,
                variant="secondary",
                size="lg",
            )

        files.change(
            fn=lambda _files: (gr.update(value=None, interactive=False), "Hazır."),
            inputs=files,
            outputs=[download, status],
            queue=False,
        )
        start.click(
            fn=_process_uploads,
            inputs=[files, preset],
            outputs=[download, status],
            concurrency_id="model-and-image-work",
            concurrency_limit=1,
            show_progress="minimal",
        )

    return demo.queue(default_concurrency_limit=1)


def main() -> None:
    create_app().launch(server_name="127.0.0.1", share=False)


if __name__ == "__main__":
    main()
