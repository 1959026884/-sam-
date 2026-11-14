# -*- coding: utf-8 -*-
"""Train the six final pair-comparator checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    from .common import (
        CHECKPOINT_DIR,
        DATA_DIR,
        DEFAULT_EPOCHS,
        ELEMENT_TO_INDEX,
        SEEDS,
        checkpoint_filename,
        set_seed,
    )
    from .models import (
        GraphormerComparator,
        WholeOnlyComparator,
        build_batch,
        graph_from_mol,
        molecule_from_smiles,
    )
except ImportError:
    from common import (
        CHECKPOINT_DIR,
        DATA_DIR,
        DEFAULT_EPOCHS,
        ELEMENT_TO_INDEX,
        SEEDS,
        checkpoint_filename,
        set_seed,
    )
    from models import (
        GraphormerComparator,
        WholeOnlyComparator,
        build_batch,
        graph_from_mol,
        molecule_from_smiles,
    )


BATCH_SIZE = 8192


def load_training_data():
    molecule_table = pd.read_excel(DATA_DIR / "合并分子清单_292.xlsx")
    if "smiles" not in molecule_table.columns:
        raise ValueError("合并分子清单缺少 smiles 列")

    molecules = sorted(
        molecule_table["smiles"].astype(str).str.strip().unique()
    )
    index = {smiles: i for i, smiles in enumerate(molecules)}

    pairs = pd.read_csv(DATA_DIR / "pseudo_pairs.csv")
    required_pair_columns = {"smiles_A", "smiles_B", "y"}
    if not required_pair_columns.issubset(pairs.columns):
        raise ValueError("pseudo_pairs.csv 缺少 smiles_A/smiles_B/y 列")

    pairs["smiles_A"] = pairs["smiles_A"].astype(str).str.strip()
    pairs["smiles_B"] = pairs["smiles_B"].astype(str).str.strip()
    unknown = (
        set(pairs["smiles_A"]) | set(pairs["smiles_B"])
    ) - set(molecules)
    if unknown:
        raise ValueError(f"配对中存在清单外分子，共 {len(unknown)} 个")

    labels = set(pairs["y"].unique().tolist())
    if not labels.issubset({0.0, 1.0}):
        raise ValueError(f"标签必须为 0/1，实际为 {sorted(labels)}")

    index_a = np.asarray([index[value] for value in pairs["smiles_A"]])
    index_b = np.asarray([index[value] for value in pairs["smiles_B"]])
    labels_tensor = torch.tensor(
        pairs["y"].to_numpy(), dtype=torch.float32
    )
    return molecules, index_a, index_b, labels_tensor


def train_wo(model, graphs, index_a, index_b, labels, epochs, tag):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=3e-3
    )
    sample_count = len(labels)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_count = 0
        permutation = torch.randperm(sample_count)
        for start in range(0, sample_count, BATCH_SIZE):
            batch_index = permutation[start:start + BATCH_SIZE]
            embeddings = model.embed_all(graphs)
            probability = model.forward_emb(
                embeddings[index_a[batch_index]],
                embeddings[index_b[batch_index]],
            )
            loss = F.binary_cross_entropy(
                probability, labels[batch_index]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_index)
            total_count += len(batch_index)
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(
                f"[{tag}] epoch {epoch}: loss={total_loss / total_count:.4f}",
                flush=True,
            )


def train_graphormer(
    model,
    batch,
    index_a,
    index_b,
    labels,
    epochs,
    tag,
):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=3e-3
    )
    sample_count = len(labels)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_count = 0
        permutation = torch.randperm(sample_count)
        for start in range(0, sample_count, BATCH_SIZE):
            batch_index = permutation[start:start + BATCH_SIZE]
            embeddings = model.embed(batch)
            probability = model.forward_emb(
                embeddings[index_a[batch_index]],
                embeddings[index_b[batch_index]],
            )
            loss = F.binary_cross_entropy(
                probability, labels[batch_index]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_index)
            total_count += len(batch_index)
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(
                f"[{tag}] epoch {epoch}: loss={total_loss / total_count:.4f}",
                flush=True,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="重训 SAM 双模型比较器")
    parser.add_argument(
        "--model", choices=["wo", "gt", "both"], default="both"
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(SEEDS)
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--outdir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖同名检查点",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="默认启用确定性算法；可用 --no-deterministic 关闭",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs 必须为正整数")
    if not args.seeds:
        parser.error("--seeds 不能为空")

    torch.set_num_threads(args.threads)
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    molecules, index_a, index_b, labels = load_training_data()
    print(f"分子={len(molecules)}, 配对={len(labels)}", flush=True)
    mols = [molecule_from_smiles(smiles) for smiles in molecules]
    graphs = [graph_from_mol(mol) for mol in mols]
    batch = build_batch(mols, ELEMENT_TO_INDEX)
    n_elements = len(ELEMENT_TO_INDEX) + 2

    selected_models = ["wo", "gt"] if args.model == "both" else [args.model]
    output_paths = [
        outdir / checkpoint_filename(model, args.epochs, seed)
        for seed in args.seeds
        for model in selected_models
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        joined = "\n".join(f"  {path}" for path in existing)
        parser.error(
            "以下检查点已存在；请更换 --outdir 或显式添加 --overwrite：\n"
            + joined
        )

    if "wo" in selected_models:
        for seed in args.seeds:
            set_seed(seed, args.deterministic)
            model = WholeOnlyComparator()
            train_wo(
                model,
                graphs,
                index_a,
                index_b,
                labels,
                args.epochs,
                f"wo_s{seed}",
            )
            output_path = outdir / checkpoint_filename(
                "wo", args.epochs, seed
            )
            torch.save(model.state_dict(), output_path)
            print(f"已保存 {output_path.name}", flush=True)

    if "gt" in selected_models:
        for seed in args.seeds:
            set_seed(seed, args.deterministic)
            model = GraphormerComparator(n_elements)
            train_graphormer(
                model,
                batch,
                index_a,
                index_b,
                labels,
                args.epochs,
                f"gt_s{seed}",
            )
            output_path = outdir / checkpoint_filename(
                "gt", args.epochs, seed
            )
            torch.save(model.state_dict(), output_path)
            print(f"已保存 {output_path.name}", flush=True)


if __name__ == "__main__":
    main()
