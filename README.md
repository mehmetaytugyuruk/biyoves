Project Website: [https://mehmetaytugyuruk.github.io/biyoves/](https://mehmetaytugyuruk.github.io/biyoves/)

## Download and run

On GitHub, select **Code → Download ZIP**. Extract the ZIP file in your Downloads folder. The extracted folder is usually named `biyoves-main`.

Follow the instructions for your computer. The first launch installs BiyoVes automatically and opens it in your browser. Keep the terminal open while using the app. Model files download automatically when you process your first photo.

### Windows

Open PowerShell and install Python 3.13:

```powershell
winget install --id Python.Python.3.13 --exact --accept-source-agreements --accept-package-agreements
```

Close and reopen PowerShell, then run BiyoVes:

```powershell
cd "$HOME\Downloads\biyoves-main"
py -3.13 start.py
```

If `winget` is not available, install Python 3.13 from [python.org](https://www.python.org/downloads/), then run the commands above.

### macOS

Open Terminal and install the `uv` tool:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close and reopen Terminal, then run BiyoVes:

```bash
cd ~/Downloads/biyoves-main
uv run --no-project --python 3.13 start.py
```

If Python is not already installed, this command downloads Python 3.13 automatically.

### Linux

Open a terminal and install the `uv` tool:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If `curl` is not available, use:

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

Close and reopen the terminal, then run BiyoVes:

```bash
cd ~/Downloads/biyoves-main
uv run --no-project --python 3.13 start.py
```

If Python is not already installed, this command downloads Python 3.13 automatically.
