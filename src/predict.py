# -*- coding: utf-8 -*-
"""Predict which of two SAM molecules is ranked higher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from .common import CHECKPOINT_DIR, DEFAULT_EPOCHS
    from .ensemble import ModelEnsemble
    from .models import molecule_from_smiles
except ImportError:
    from common import CHECKPOINT_DIR, DEFAULT_EPOCHS
    from ensemble import ModelEnsemble
    from models import molecule_from_smiles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="比较两个 SAM 分子的相对排序"
    )
    parser.add_argument("smiles_a", help="第一个分子的 SMILES")
    parser.add_argument("smiles_b", help="第二个分子的 SMILES")
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=CHECKPOINT_DIR
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--json",
        action="store_true",
        help="仅输出 JSON，便于程序调用",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    try:
        molecule_from_smiles(args.smiles_a)
        molecule_from_smiles(args.smiles_b)
        ensemble = ModelEnsemble(args.checkpoint_dir, epochs=args.epochs)
        probability, p_wo, p_gt = ensemble.predict_pairs(
            [args.smiles_a, args.smiles_b],
            [args.smiles_b, args.smiles_a],
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    p_a_b = float(probability[0])
    p_b_a = float(probability[1])
    result = {
        "p_a_better_than_b": p_a_b,
        "p_b_better_than_a": p_b_a,
        "direction_sum": p_a_b + p_b_a,
        "whole_only_a_better_than_b": float(p_wo[0]),
        "graphormer_a_better_than_b": float(p_gt[0]),
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(
        f"P(A 优于 B) = {p_a_b:.4f}   "
        f"(整分子图 {p_wo[0]:.4f} | Graphormer {p_gt[0]:.4f})"
    )
    print(
        f"反方向 P(B 优于 A) = {p_b_a:.4f}   "
        f"两者之和 = {p_a_b + p_b_a:.4f}（应≈1.0000）"
    )


if __name__ == "__main__":
    main()
