import os
from pathlib import Path

import pytest
from PIL import Image

from biyoves.geometry import decode_image, write_output
from biyoves.runtime import ProductionPipeline, download_models


@pytest.mark.smoke
def test_production_pipeline_smoke(tmp_path):
    source = tmp_path / "portrait.png"
    showcase = Path(__file__).parents[1] / "docs" / "assets" / "readme" / "showcase.webp"
    with Image.open(showcase) as sheet:
        sheet.crop((30, 25, 302, 385)).convert("RGB").save(source)

    cache = Path(os.environ.get("BIYOVES_CACHE_DIR", str(tmp_path / "cache")))
    paths = download_models(cache)
    assert paths.birefnet_snapshot.is_dir()
    assert paths.face_landmarker.is_file()

    with ProductionPipeline(cache) as pipeline:
        output, diagnostics = pipeline.process(
            decode_image(source),
            dpi=300,
            preset="biometric",
        )

    output_path = tmp_path / "output.png"
    write_output(output_path, output, dpi=300)
    with Image.open(output_path) as result:
        assert result.size == (591, 709)
        assert result.mode == "RGB"
    assert diagnostics["inference"]["network"] is False
    assert diagnostics["model"]["revision"]
