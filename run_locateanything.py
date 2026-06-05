#!/usr/bin/env python3
"""Minimal Apple Silicon runner for nvidia/LocateAnything-3B."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HOME", str(Path(".hf-cache").resolve()))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoProcessor, AutoTokenizer


MODEL_ID = "nvidia/LocateAnything-3B"


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def choose_dtype(name: str, device: str) -> torch.dtype:
    if name == "auto":
        if device == "cpu":
            return torch.float32
        return torch.float16
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


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
    parser.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--generation-mode", choices=["slow", "hybrid", "fast"], default="slow")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--online", action="store_true", help="Allow Hugging Face network checks/downloads.")
    args = parser.parse_args()

    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)
    image_path = args.image or make_test_image(Path("test_image.png"))
    image = Image.open(image_path).convert("RGB")
    prompt = build_prompt(args.task, args.query)

    print(f"device={device} dtype={dtype} generation_mode={args.generation_mode}")
    print(f"image={image_path} size={image.size}")
    print(f"prompt={prompt}")

    local_files_only = not args.online
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="sdpa",
        local_files_only=local_files_only,
    ).to(device).eval()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = processor.process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to(device)

    with torch.inference_mode():
        response = model.generate(
            pixel_values=inputs["pixel_values"].to(dtype),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws"),
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            generation_mode=args.generation_mode,
            temperature=args.temperature,
            do_sample=args.temperature > 0,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=True,
        )

    answer = response[0] if isinstance(response, tuple) else response
    print("\nanswer:")
    print(answer)
    print("\nboxes:")
    print(parse_boxes(answer, *image.size))
    print("\npoints:")
    print(parse_points(answer, *image.size))


if __name__ == "__main__":
    main()
