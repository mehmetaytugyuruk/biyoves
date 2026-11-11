"""Failure paths for the shared model cache and first-run downloads."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import biyoves.runtime as runtime
from biyoves.errors import ModelAssetsError


def test_abandoned_model_lock_does_not_block_installation(tmp_path):
    (tmp_path / ".models.lock").write_text("old process")
    with runtime._cache_lock(tmp_path):
        assert (tmp_path / ".models.lock").is_file()
    with runtime._cache_lock(tmp_path):
        pass


def test_model_lock_serializes_contenders(tmp_path, monkeypatch):
    with runtime._cache_lock(tmp_path):
        times = iter([0, 301])
        monkeypatch.setattr(runtime.time, "monotonic", lambda: next(times))
        with pytest.raises(ModelAssetsError, match="timed out"):
            with runtime._cache_lock(tmp_path):
                pytest.fail("contender entered the critical section")


def test_model_lock_released_after_process_is_killed(tmp_path):
    environment = dict(os.environ, PYTHONPATH=str(Path(runtime.__file__).parents[1]))
    code = (
        "import sys,time; from pathlib import Path; from biyoves.runtime import _cache_lock; "
        "lock=_cache_lock(Path(sys.argv[1])); lock.__enter__(); "
        "print('locked', flush=True); time.sleep(60)"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    try:
        assert child.stdout.readline().strip() == "locked"
    finally:
        child.kill()
        child.communicate(timeout=15)
    subprocess.run(
        [sys.executable, "-c", code.replace("time.sleep(60)", "lock.__exit__(None,None,None)"),
         str(tmp_path)],
        env=environment, check=True, capture_output=True, timeout=15,
    )


def test_interrupted_asset_download_preserves_previous_file(tmp_path, monkeypatch):
    destination = tmp_path / "model.task"
    destination.write_bytes(b"previous verified model")

    def disconnected(*args, **kwargs):
        raise OSError("connection lost")

    monkeypatch.setattr(runtime.urllib.request, "urlopen", disconnected)
    with pytest.raises(OSError, match="connection lost"):
        runtime._download_url("https://example.invalid/model", destination)
    assert destination.read_bytes() == b"previous verified model"
    assert list(tmp_path.iterdir()) == [destination]
