"""Create the project environment on first run, then start BiyoVes."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import venv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_DIR = PROJECT_ROOT / ".venv"
INSTALL_MARKER = ENV_DIR / ".biyoves-install-key"


def environment_python() -> Path:
    if os.name == "nt":
        return ENV_DIR / "Scripts" / "python.exe"
    return ENV_DIR / "bin" / "python"


def installation_key() -> str:
    return hashlib.sha256((PROJECT_ROOT / "pyproject.toml").read_bytes()).hexdigest()


def ensure_environment() -> Path:
    if not (3, 10) <= sys.version_info[:2] < (3, 14):
        raise SystemExit("BiyoVes needs Python 3.10–3.13. Run start.py with a supported Python.")

    python = environment_python()
    if not python.is_file():
        if ENV_DIR.exists():
            raise SystemExit(
                f"The existing environment at {ENV_DIR} is incomplete. "
                "Remove that .venv directory and run this command again."
            )
        print("Creating the project environment (.venv)…", flush=True)
        venv.create(ENV_DIR, with_pip=True)

    key = installation_key()
    if not INSTALL_MARKER.is_file() or INSTALL_MARKER.read_text(encoding="utf-8").strip() != key:
        print("Installing BiyoVes and its Python dependencies…", flush=True)
        subprocess.run(
            [str(python), "-m", "pip", "install", "--disable-pip-version-check", "-e", "."],
            cwd=PROJECT_ROOT,
            check=True,
        )
        INSTALL_MARKER.write_text(key + "\n", encoding="utf-8")

    return python


def main() -> int:
    python = ensure_environment()
    return subprocess.run([str(python), "-m", "biyoves"], cwd=PROJECT_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
