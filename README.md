# Locate Anything GUI

This is a native macOS Qt webcam GUI for `nvidia/LocateAnything-3B`, using the Apple Silicon MLX quantization `mlx-community/LocateAnything-3B-4bit`.

The main app runs the MLX 4-bit model on Metal and keeps it loaded in memory for semi-real-time webcam demos. The original PyTorch/MPS scripts are kept as compatibility experiments, but the recommended Mac path is the MLX GUI.

## Setup

```bash
UV_CACHE_DIR=.uv-cache uv sync --python 3.12
```

The MLX LocateAnything support currently lives on an `mlx-vlm` branch whose dependency metadata conflicts with released `transformers`, so install that branch without dependency resolution:

```bash
UV_CACHE_DIR=.uv-cache uv pip install --no-deps mlx-lm==0.31.3 "git+https://github.com/beshkenadze/mlx-vlm@feat/locateanything-3b"
```

## PyTorch/MPS Fallback

The original NVIDIA model can also be tested through PyTorch/MPS, but that path is slower and may fall back to CPU for unsupported operations. Download the original model first. This is about 8 GB and is resumable.

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

This uses `mlx-community/LocateAnything-3B-4bit` and downloads about 3 GB on first run. That model is an MLX quantization intended for Apple Silicon/macOS rather than a portable CUDA/GGUF format.

## Native Qt Webcam Demo

```bash
UV_CACHE_DIR=.uv-cache uv run python qt_realtime_locate.py
```

The app opens camera index `0`, keeps the MLX 4-bit model loaded in a worker thread, shows a live preview, and runs recognition as fast as the model finishes each frame. The default settings use `hybrid` mode, 640x360 camera capture, inference width `256`, and max tokens `64`. Use:

```bash
UV_CACHE_DIR=.uv-cache uv run python qt_realtime_locate.py --camera 1
```

if your webcam is not index `0`. macOS may ask for camera permission for the Python process the first time.

On a MacBook Air M3, the MLX 4-bit path runs semi-real-time for webcam demos with the default low-resolution inference settings. It is not a temporal video tracker; it processes the latest camera frame as an image and immediately schedules the next frame when inference finishes.

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
- This repository's code is licensed under GPL-3.0-only. The upstream model weights and third-party dependencies keep their own licenses.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set by the script before importing torch.
- `HF_HOME=.hf-cache` is set by the scripts so model files stay inside this project.
- `HF_HUB_DISABLE_XET=1` is set by the scripts to force standard Hugging Face HTTPS downloads on macOS.
- If MPS hits an unsupported operation, PyTorch can fall back to CPU for that operation.
- If you run out of memory, use a smaller input image, reduce `--max-new-tokens`, or try `--device cpu --dtype float32`.
