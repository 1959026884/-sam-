# -*- coding: utf-8 -*-
"""Evaluate reference rankings and optionally rank a candidate library."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

try:
    from .common import (
        CHECKPOINT_DIR,
        DEFAULT_EPOCHS,
        RESULTS_DIR,
    )
    from .ensemble import ModelEnsemble
except ImportError:
    from common import (
        CHECKPOINT_DIR,
        DEFAULT_EPOCHS,
        RESULTS_DIR,
    )
    from ensemble import ModelEnsemble


REFERENCES = {
    "4PACz": "O=P(O)(O)CCCCn1c2ccccc2c2ccccc21",
    "MeO-2PACz": "COc1ccc2c(c1)c1cc(OC)ccc1n2CCP(=O)(O)O",
    "2PACz": "O=P(O)(O)CCn1c2ccccc2c2ccccc21",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="评估基准分子排序并可选执行候选初筛"
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=CHECKPOINT_DIR
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--screen", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--threads", type=int, default=4)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    results_dir = args.results_dir.expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    ensemble = ModelEnsemble(args.checkpoint_dir, epochs=args.epochs)

    names = list(REFERENCES)
    reference_smiles = [REFERENCES[name] for name in names]
    matrix_a = [
        smiles_a for smiles_a in reference_smiles for _ in names
    ]
    matrix_b = [
        smiles_b for _ in names for smiles_b in reference_smiles
    ]
    matrix_probability, _, _ = ensemble.predict_pairs(matrix_a, matrix_b)
    probability_matrix = matrix_probability.reshape(
        len(names), len(names)
    )

    antisymmetry = float(
        probability_matrix[0, 1] + probability_matrix[1, 0]
    )
    print(
        "反称性: "
        f"P(4PACz>MeO-2PACz)={probability_matrix[0, 1]:.4f} + "
        f"P(MeO-2PACz>4PACz)={probability_matrix[1, 0]:.4f} "
        f"= {antisymmetry:.4f}"
    )

    print("\n基准分子两两排序（P(行 优于 列)）:")
    print("         " + "  ".join(f"{name:>10s}" for name in names))
    for row_index, name in enumerate(names):
        values = "  ".join(
            f"{value:.4f}" for value in probability_matrix[row_index]
        )
        print(f"{name:>8s}  {values}")

    matrix_table = pd.DataFrame(
        probability_matrix, index=names, columns=names
    )
    matrix_table.index.name = "row_better_than_column"
    matrix_table.to_csv(
        results_dir / "baseline_pair_matrix.csv",
        encoding="utf-8-sig",
    )

    metrics = {
        "antisymmetry_sum": antisymmetry,
        "reference_order": names,
        "reference_pair_matrix": {
            row_name: {
                column_name: float(probability_matrix[row_index, column_index])
                for column_index, column_name in enumerate(names)
            }
            for row_index, row_name in enumerate(names)
        },
    }

    if args.screen:
        screen = pd.read_csv(args.screen)
        if "smiles" not in screen.columns:
            parser.error("--screen CSV 必须包含 smiles 列")
        candidates = screen["smiles"].astype(str).str.strip().tolist()
        if not candidates:
            parser.error("--screen CSV 中没有候选分子")
        names_for_output = []
        for index, row in enumerate(screen.to_dict(orient="records")):
            if "name" in screen.columns and pd.notna(row["name"]):
                names_for_output.append(str(row["name"]))
            else:
                names_for_output.append(candidates[index])

        screen_a = [
            candidate
            for candidate in candidates
            for _ in reference_smiles
        ]
        screen_b = reference_smiles * len(candidates)
        try:
            screen_probability, _, _ = ensemble.predict_pairs(
                screen_a, screen_b
            )
        except ValueError as exc:
            parser.error(str(exc))
        mean_win = screen_probability.reshape(
            len(candidates), len(reference_smiles)
        ).mean(axis=1)
        ranking = pd.DataFrame(
            {
                "name": names_for_output,
                "smiles": candidates,
                "mean_win_vs_references": mean_win,
            }
        ).sort_values(
            "mean_win_vs_references", ascending=False
        )
        screen_output = results_dir / "screen_rank.csv"
        ranking.to_csv(
            screen_output, index=False, encoding="utf-8-sig"
        )
        metrics["screen_candidates"] = int(len(ranking))
        print(f"\n初筛完成: {len(ranking)} 个分子 -> {screen_output}")
        print(ranking.head(10).round(4).to_string(index=False))

    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n结果已写入: {results_dir}")


if __name__ == "__main__":
    main()
