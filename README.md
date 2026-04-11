New to the repo? Read **[CODEBASE_GUIDE.md](CODEBASE_GUIDE.md)** for architecture, data contracts, pitfalls, and next steps.
 
1. Create the virtual environment
python3 -m venv .venv

2. Activate it
On Windows:
```bash
.\.venv\Scripts\activate
```

On macOS/Linux:
```bash
source .venv/bin/activate
```

3. Install requirements
```bash
pip install -r requirements.txt
```

4. Verify setup (after adding `.env` with `XC_API_KEY` for API v3)
```bash
python scripts/check_environment.py
python scripts/check_environment.py --with-ml
```

5. Optional: CLAP smoke test (installs PyTorch stack; MP3 decoding usually needs [ffmpeg](https://ffmpeg.org/download.html) on PATH)
```bash
pip install -r requirements-ml.txt
python scripts/mini_clap_xc_sample.py --sample 6
```


