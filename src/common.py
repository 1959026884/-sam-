# -*- coding: utf-8 -*-
"""Shared paths, constants, and reproducibility helpers."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

SEEDS = (0, 1, 2)
DEFAULT_EPOCHS = 600
MODEL_NAMES = ("wo", "gt")

ELEMENTS = (
    "B", "Br", "C", "Cl", "F", "H", "I",
    "N", "O", "P", "S", "Se", "Si",
)
ELEMENT_TO_INDEX = {element: index + 2 for index, element in enumerate(ELEMENTS)}
SUPPORTED_ELEMENTS = frozenset(ELEMENTS)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed all libraries used for initialization and minibatch shuffling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)


def checkpoint_filename(model: str, epochs: int, seed: int) -> str:
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model}")
    return f"full292_{model}_x{epochs}_s{seed}.pt"


def checkpoint_path(
    checkpoint_dir: str | Path,
    model: str,
    epochs: int,
    seed: int,
) -> Path:
    return Path(checkpoint_dir).expanduser().resolve() / checkpoint_filename(
        model, epochs, seed
    )
