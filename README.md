# LocateAnything-3B on Apple Silicon

This is a minimal Mac test harness for `nvidia/LocateAnything-3B`.

The model is built for CUDA/Linux first, so the first runnable Mac path uses PyTorch MPS with CPU fallback enabled. Expect partial acceleration, not CUDA-class throughput.

## Setup

```bash
UV_CACHE_DIR=.uv-cache uv sync --python 3.12
```

The MLX LocateAnything support currently lives on an `mlx-vlm` branch whose dependency metadata conflicts with released `transformers`, so install that branch without dependency resolution:

```bash
UV_CACHE_DIR=.uv-cache uv pip install --no-deps mlx-lm==0.31.3 "git+https://github.com/beshkenadze/mlx-vlm@feat/locateanything-3b"
```

## First Run

Download the model first. This is about 8 GB and is resumable.

```bash
UV_CACHE_DIR=.uv-cache uv run python download_model.py
```

If that stalls, download only the two large weight shards:

```bash
UV_CACHE_DIR=.uv-cache uv run python download_weights.py
```

Then run the smoke test:

```bash
UV_CACHE_DIR=.uv-cache uv run python run_locateanything.py --task point --query "the red square" --max-new-tokens 32
```

The script creates a simple local test image if you do not pass `--image`.

## Useful Options

```bash
UV_CACHE_DIR=.uv-cache uv run python run_locateanything.py --image path/to/image.jpg --task detect --query "person, car"
UV_CACHE_DIR=.uv-cache uv run python run_locateanything.py --image path/to/screen.png --task gui-point --query "the search button"
UV_CACHE_DIR=.uv-cache uv run python run_locateanything.py --device cpu --dtype float32
UV_CACHE_DIR=.uv-cache uv run python run_locateanything.py --generation-mode hybrid --max-new-tokens 2048
```

Start with `--generation-mode slow` for the original PyTorch/MPS path if you are debugging compatibility. It is slower, but avoids the fast parallel decoding path while proving the model runs.

## MLX 4-bit Run

The MLX 4-bit conversion is much faster on Apple Silicon:

```bash
UV_CACHE_DIR=.uv-cache uv run python run_locateanything_mlx.py --task point --query "the red square"
```

This uses `mlx-community/LocateAnything-3B-4bit` and downloads about 3 GB on first run.

## Native Qt Webcam Demo

```bash
UV_CACHE_DIR=.uv-cache uv run python qt_realtime_locate.py
```

The app opens camera index `0`, keeps the MLX 4-bit model loaded in a worker thread, shows a live preview, and runs recognition as fast as the model finishes each frame. The default settings use `hybrid` mode, 640x360 camera capture, inference width `256`, and max tokens `64`. Use:

```bash
UV_CACHE_DIR=.uv-cache uv run python qt_realtime_locate.py --camera 1
```

if your webcam is not index `0`. macOS may ask for camera permission for the Python process the first time.

Controls:

- `Start camera`: starts or stops live webcam preview.
- `Upload image`: loads a still image, pauses live inference, and lets you use `Run now`.
- `Pause`: stops continuous inference and reveals `Run now`.
- `Run now`: runs one inference on the current camera frame or uploaded image; only shown while paused.
- `Inference width`: resizes the image before inference; `128` and `192` are low-quality speed tests, `256` is the default, and `0` uses the full frame.

Task modes change how the textbox is turned into a model prompt:

- `detect`: textbox is a comma-separated category list, such as `cup, keyboard`; returns boxes for matching objects.
- `point`: textbox is a referring expression, such as `the red mug`; returns a point/crosshair.
- `ground-single`: returns one region matching a referring expression.
- `ground-multi`: returns all regions matching a referring expression.
- `gui-box`: for screenshots or UI images; returns a box for a UI element, such as `the search button`.
- `gui-point`: for screenshots or UI images; returns a point for a UI element, such as `the close icon`.

For webcam scenes, start with `detect`, `point`, `ground-single`, or `ground-multi`. The `gui-*` modes are mainly for screen/UI images.

## Notes

- The model license is non-commercial research use only.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set by the script before importing torch.
- `HF_HOME=.hf-cache` is set by the scripts so model files stay inside this project.
- `HF_HUB_DISABLE_XET=1` is set by the scripts to force standard Hugging Face HTTPS downloads on macOS.
- If MPS hits an unsupported operation, PyTorch can fall back to CPU for that operation.
- If you run out of memory, use a smaller input image, reduce `--max-new-tokens`, or try `--device cpu --dtype float32`.
