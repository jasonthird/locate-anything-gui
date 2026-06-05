#!/usr/bin/env python3
"""Download LocateAnything-3B into the local Hugging Face cache."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(".hf-cache").resolve()))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download


snapshot_download(
    repo_id="nvidia/LocateAnything-3B",
    resume_download=True,
    max_workers=1,
)
