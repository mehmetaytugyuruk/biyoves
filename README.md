# BiyoVes

BiyoVes is a local web app for preparing biometric, vesikalık, and visa photos. It processes
photos on your computer and creates three quality variants, each available as a single photo
or a 10 × 15 cm print sheet.

## Requirements

- Python 3.10–3.13
- macOS Apple Silicon, Windows x64, or Linux x64
- An internet connection on first launch to install dependencies, and on first photo processing to download the model files

## Install and run

Download the repository as a ZIP from GitHub (**Code → Download ZIP**) and extract it. GitHub names the extracted folder `biyoves-main`; open a terminal in that folder before running the launcher. If you already use Git, clone the repository and enter its `biyoves` folder with the commands below. On first launch the app creates an isolated `.venv`, installs dependencies, and starts the local server. Run the same command again to start BiyoVes later; no environment activation is needed.

Git is optional. Python itself must be installed separately; BiyoVes supports Python 3.10–3.13. On Windows, install Python 3.13 (64-bit) and use the version-specific command below so another installed Python version is not selected by mistake.

On Windows 10/11 with WinGet available, Python can be installed from PowerShell without manually downloading the installer:

```powershell
winget install --id Python.Python.3.13 --exact --accept-source-agreements --accept-package-agreements
```

Reopen the terminal after installation, then run BiyoVes with `py -3.13 start.py`. WinGet may still display an installation or confirmation prompt. If `winget` is unavailable, install Python 3.13 (64-bit) from [python.org](https://www.python.org/downloads/release/python-31316/).

```bash
git clone https://github.com/mehmetaytugyuruk/biyoves.git
cd biyoves
```

On macOS or Linux:

```bash
python3 start.py
```

On Windows:

```powershell
py -3.13 start.py
```

Open <http://127.0.0.1:7860> in your browser. Model files are downloaded and verified when the first photo is processed, then reused from the local cache.

If port 7860 is already in use, choose another port when starting the app:

```bash
BIYOVES_PORT=7861 python3 start.py
```

```powershell
$env:BIYOVES_PORT = "7861"
py -3.13 start.py
```

## Photo formats

| Format | Canvas | Visible head target | Top margin |
| --- | ---: | ---: | ---: |
| Biometric | 50 × 60 mm | 36 mm | 3 mm |
| Vesikalık | 45 × 60 mm | 32 mm | 3 mm |
| USA visa | 50 × 50 mm | 36 mm | 1 mm |
| Schengen visa | 35 × 45 mm | 36 mm | 1 mm |

## 10 × 15 cm print sheets

Each output variant also includes a 10 × 15 cm portrait PNG with four copies of
the selected photo at its preset's physical size. Copies have a 2 mm gap where
the sheet dimensions allow it; biometric and USA visa copies touch horizontally
because each is 50 mm wide. A gray cutting frame is included around each copy.
When printing, use 10 × 15 cm paper and select 100% / actual size so the photo
dimensions are preserved.

## License

The source code is released under the MIT License. BiRefNet Lite and MediaPipe assets remain subject to their upstream licenses and terms.
