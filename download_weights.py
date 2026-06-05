#!/usr/bin/env python3
"""Download only the LocateAnything-3B weight shards."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(".hf-cache").resolve()))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import hf_hub_download


REPO_ID = "nvidia/LocateAnything-3B"
WEIGHT_FILES = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
]


for filename in WEIGHT_FILES:
    print(f"downloading {filename}", flush=True)
    path = hf_hub_download(repo_id=REPO_ID, filename=filename, resume_download=True)
    print(f"downloaded {filename}: {path}", flush=True)
