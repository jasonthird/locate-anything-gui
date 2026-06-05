#!/usr/bin/env python3
"""MLX runner for mlx-community/LocateAnything-3B-4bit."""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(".hf-cache").resolve()))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from PIL import Image, ImageDraw
import transformers
import transformers.processing_utils as transformers_processing_utils
from mlx_vlm.models.locateanything.image_processing_locateanything import (
    LocateAnythingImageProcessor,
)
from mlx_vlm.models.locateanything.processing_locateanything import (
    LocateAnythingProcessor,
)
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import get_model_path, load_model, prepare_inputs


MODEL_ID = "mlx-community/LocateAnything-3B-4bit"
transformers.LocateAnythingProcessor = LocateAnythingProcessor
transformers.LocateAnythingImageProcessor = LocateAnythingImageProcessor
transformers_processing_utils.transformers_module.LocateAnythingProcessor = LocateAnythingProcessor
transformers_processing_utils.transformers_module.LocateAnythingImageProcessor = LocateAnythingImageProcessor


def make_test_image(path: Path) -> Path:
    image = Image.new("RGB", (640, 420), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((90, 90, 260, 250), fill="red")
    draw.ellipse((360, 115, 540, 295), fill="royalblue")
    draw.text((110, 280), "red square", fill="black")
    draw.text((385, 320), "blue circle", fill="black")
    image.save(path)
    return path


def build_prompt(task: str, query: str) -> str:
    if task == "detect":
        categories = "</c>".join(part.strip() for part in query.split(",") if part.strip())
        return f"Locate all the instances that matches the following description: {categories}."
    if task == "ground-single":
        return f"Locate a single instance that matches the following description: {query}."
    if task == "ground-multi":
        return f"Locate all the instances that match the following description: {query}."
    if task == "text":
        return "Detect all the text in box format." if not query else f"Please locate the text referred as {query}."
    if task == "gui-box":
        return f"Locate the region that matches the following description: {query}."
    if task in {"point", "gui-point"}:
        return f"Point to: {query}."
    raise ValueError(f"Unsupported task: {task}")


def parse_boxes(answer: str, width: int, height: int) -> list[dict[str, float]]:
    boxes = []
    for match in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
        x1, y1, x2, y2 = [int(group) for group in match.groups()]
        boxes.append(
            {
                "x1": x1 / 1000 * width,
                "y1": y1 / 1000 * height,
                "x2": x2 / 1000 * width,
                "y2": y2 / 1000 * height,
            }
        )
    return boxes


def parse_points(answer: str, width: int, height: int) -> list[dict[str, float]]:
    points = []
    for match in re.finditer(r"<box><(\d+)><(\d+)></box>", answer):
        x, y = [int(group) for group in match.groups()]
        points.append({"x": x / 1000 * width, "y": y / 1000 * height})
    return points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--query", default="the red square")
    parser.add_argument(
        "--task",
        choices=["detect", "ground-single", "ground-multi", "text", "gui-box", "gui-point", "point"],
        default="point",
    )
    parser.add_argument("--generation-mode", choices=["slow", "hybrid", "fast"], default="hybrid")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    image_path = args.image or make_test_image(Path("test_image.png"))
    image = Image.open(image_path).convert("RGB")
    prompt_text = build_prompt(args.task, args.query)

    print(f"model={args.model}")
    print(f"generation_mode={args.generation_mode} max_tokens={args.max_tokens}")
    print(f"image={image_path} size={image.size}")
    print(f"prompt={prompt_text}")

    load_start = time.perf_counter()
    model_path = get_model_path(args.model)
    model = load_model(model_path)
    processor = LocateAnythingProcessor.from_pretrained(model_path)
    load_time = time.perf_counter() - load_start

    prompt = apply_chat_template(processor, model.config, prompt_text, num_images=1)
    inputs = prepare_inputs(processor, images=[str(image_path)], prompts=prompt)
    input_ids = inputs.pop("input_ids")
    inputs.pop("attention_mask", None)

    gen_start = time.perf_counter()
    tokens = model.pbd_generate(
        input_ids,
        generation_mode=args.generation_mode,
        max_tokens=args.max_tokens,
        **inputs,
    )
    gen_time = time.perf_counter() - gen_start

    answer = processor.decode(tokens, skip_special_tokens=False)
    token_count = len(tokens)
    print(f"\nload_time_s={load_time:.4f}")
    print(f"generate_time_s={gen_time:.4f}")
    print(f"tokens={token_count}")
    if gen_time > 0:
        print(f"tokens_per_s={token_count / gen_time:.4f}")

    print("\nanswer:")
    print(answer)
    print("\nboxes:")
    print(parse_boxes(answer, *image.size))
    print("\npoints:")
    print(parse_points(answer, *image.size))


if __name__ == "__main__":
    main()
