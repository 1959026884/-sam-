# -*- coding: utf-8 -*-
"""Leakage-safe five-fold training with Optuna hyperparameter tuning."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from chemprop.data.collate import BatchMolGraph
from optuna.exceptions import TrialPruned
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from rdkit import Chem
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

try:
    from .common import DATA_DIR, DEFAULT_EPOCHS, ELEMENT_TO_INDEX, SEEDS
    from .models import (
        GraphormerComparator,
        WholeOnlyComparator,
        build_batch,
        graph_from_mol,
        molecule_from_smiles,
    )
except ImportError:
    from common import DATA_DIR, DEFAULT_EPOCHS, ELEMENT_TO_INDEX, SEEDS
    from models import (
        GraphormerComparator,
        WholeOnlyComparator,
        build_batch,
        graph_from_mol,
        molecule_from_smiles,
    )


N_FOLDS = 5
MODEL_NAMES = ("wo", "gt")
MODEL_LABELS = {"wo": "wo", "gt": "gt"}


@dataclass
class Dataset:
    smiles: list[str]
    molecules: list[Chem.Mol]
    pce: np.ndarray
    pair_i: np.ndarray
    pair_j: np.ndarray
    pair_y: np.ndarray
    wo_graphs: BatchMolGraph
    gt_batch: tuple[torch.Tensor, ...]
    n_elements: int


@dataclass
class PairSplit:
    train_i: torch.Tensor
    train_j: torch.Tensor
    train_y: torch.Tensor
    val_i: torch.Tensor
    val_j: torch.Tensor
    val_y: torch.Tensor
    strict_val_mask: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAM 分子隔离 5 折训练、Optuna 调参与最终重训"
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES)
    )
    parser.add_argument("--trials-per-fold", type=int, default=8)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=20261113)
    parser.add_argument("--tuning-seed", type=int, default=20261114)
    parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu"], default="auto"
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("runs") / f"sam_cv_{datetime.now():%Y%m%d_%H%M%S}",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-tuning", action="store_true")
    parser.add_argument("--tune-only", action="store_true")
    parser.add_argument("--skip-final-full", action="store_true")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def configure_logging(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "train.log"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("optuna").setLevel(logging.WARNING)


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        raise RuntimeError("请求了 CUDA，但当前 PyTorch 无法访问 GPU")
    return torch.device("cpu")


def release_accelerator_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def load_dataset() -> Dataset:
    molecule_table = pd.read_excel(DATA_DIR / "合并分子清单_292.xlsx")
    if "smiles" not in molecule_table.columns:
        raise ValueError("合并分子清单缺少 smiles 列")
    if "PCE_max" not in molecule_table.columns:
        raise ValueError("合并分子清单缺少 PCE_max 列")

    frame = molecule_table[["smiles", "PCE_max"]].copy()
    frame["smiles"] = frame["smiles"].astype(str).str.strip()
    frame = frame.drop_duplicates(subset="smiles").sort_values("smiles")
    smiles = frame["smiles"].tolist()
    pce = pd.to_numeric(frame["PCE_max"], errors="raise").to_numpy(
        dtype=np.float64
    )
    molecules = [molecule_from_smiles(value) for value in smiles]

    canonical = [
        Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        for mol in molecules
    ]
    if len(set(canonical)) != len(canonical):
        duplicates = [
            value for value, count in Counter(canonical).items() if count > 1
        ]
        raise ValueError(
            "训练清单包含规范化后重复的分子，请先去重: "
            + ", ".join(duplicates[:5])
        )

    raw_pairs = pd.read_csv(DATA_DIR / "pseudo_pairs.csv")
    required = {"smiles_A", "smiles_B", "y"}
    if not required.issubset(raw_pairs.columns):
        raise ValueError("pseudo_pairs.csv 缺少 smiles_A/smiles_B/y 列")

    index = {value: i for i, value in enumerate(smiles)}
    grouped: dict[tuple[str, str], dict[str, int]] = {}
    for row in raw_pairs.itertuples(index=False):
        a = str(row.smiles_A).strip()
        b = str(row.smiles_B).strip()
        if a not in index or b not in index:
            raise ValueError("配对中出现训练清单外分子")
        if a == b:
            raise ValueError("配对中出现自配对")
        label = int(row.y)
        if label not in {0, 1}:
            raise ValueError("配对标签必须为 0 或 1")
        key = tuple(sorted((a, b)))
        grouped.setdefault(key, {})[a] = label

    pair_i: list[int] = []
    pair_j: list[int] = []
    pair_y: list[int] = []
    for (a, b), directions in sorted(grouped.items()):
        if set(directions) != {a, b}:
            raise ValueError(f"配对缺少双向记录: {a} | {b}")
        if directions[a] + directions[b] != 1:
            raise ValueError(f"配对双向标签不互补: {a} | {b}")
        pair_i.append(index[a])
        pair_j.append(index[b])
        pair_y.append(directions[a])

    pair_i_np = np.asarray(pair_i, dtype=np.int64)
    pair_j_np = np.asarray(pair_j, dtype=np.int64)
    pair_y_np = np.asarray(pair_y, dtype=np.float32)

    graphs = BatchMolGraph([graph_from_mol(mol) for mol in molecules])
    gt_batch = build_batch(molecules, ELEMENT_TO_INDEX)
    return Dataset(
        smiles=smiles,
        molecules=molecules,
        pce=pce,
        pair_i=pair_i_np,
        pair_j=pair_j_np,
        pair_y=pair_y_np,
        wo_graphs=graphs,
        gt_batch=gt_batch,
        n_elements=len(ELEMENT_TO_INDEX) + 2,
    )


def assign_molecule_folds(
    dataset: Dataset,
    n_folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Balance molecule counts and PCE quantiles across folds."""
    rng = np.random.default_rng(seed)
    fold_of = np.full(len(dataset.smiles), -1, dtype=np.int64)
    pce_order = np.argsort(dataset.pce, kind="stable")
    quantile_bins = np.array_split(pce_order, n_folds * 4)
    fold_sizes = np.zeros(n_folds, dtype=np.int64)
    for bin_index, members in enumerate(quantile_bins):
        members = rng.permutation(members)
        preferred = bin_index % n_folds
        for offset, molecule_index in enumerate(members):
            candidates = [
                (fold_sizes[(preferred + step) % n_folds],
                 (preferred + step) % n_folds)
                for step in range(n_folds)
            ]
            _, fold = min(candidates)
            fold_of[int(molecule_index)] = fold
            fold_sizes[fold] += 1

    # Every edge selects the fold of its validation target. This keeps the
    # partner known to the model while ensuring the target molecule is unseen.
    pair_fold = np.empty(len(dataset.pair_i), dtype=np.int64)
    target_side = np.empty(len(dataset.pair_i), dtype=np.int64)
    validation_load = np.zeros(n_folds, dtype=np.int64)
    for pair_index, (a, b) in enumerate(zip(dataset.pair_i, dataset.pair_j)):
        fold_a = int(fold_of[a])
        fold_b = int(fold_of[b])
        if fold_a == fold_b:
            fold = fold_a
            side = 0
        elif validation_load[fold_a] < validation_load[fold_b]:
            fold = fold_a
            side = 0
        elif validation_load[fold_b] < validation_load[fold_a]:
            fold = fold_b
            side = 1
        elif int(a) <= int(b):
            fold = fold_a
            side = 0
        else:
            fold = fold_b
            side = 1
        pair_fold[pair_index] = fold
        target_side[pair_index] = side
        validation_load[fold] += 1

    if np.any(fold_of < 0):
        raise RuntimeError("存在未分配到折的分子")
    return fold_of, pair_fold, target_side


def to_device_dataset(dataset: Dataset, device: torch.device) -> None:
    dataset.wo_graphs.to(device)
    dataset.gt_batch = tuple(tensor.to(device) for tensor in dataset.gt_batch)


def tensor_indices(
    i: np.ndarray,
    j: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(i, dtype=torch.long, device=device),
        torch.as_tensor(j, dtype=torch.long, device=device),
        torch.as_tensor(y, dtype=torch.float32, device=device),
    )


def build_outer_split(
    dataset: Dataset,
    fold_of: np.ndarray,
    pair_fold: np.ndarray,
    target_side: np.ndarray,
    fold: int,
    device: torch.device,
) -> PairSplit:
    heldout = fold_of == fold
    train_mask = ~(heldout[dataset.pair_i] | heldout[dataset.pair_j])
    val_mask = pair_fold == fold

    val_a = np.where(
        target_side[val_mask] == 0,
        dataset.pair_i[val_mask],
        dataset.pair_j[val_mask],
    )
    val_b = np.where(
        target_side[val_mask] == 0,
        dataset.pair_j[val_mask],
        dataset.pair_i[val_mask],
    )
    val_y = np.where(
        target_side[val_mask] == 0,
        dataset.pair_y[val_mask],
        1.0 - dataset.pair_y[val_mask],
    )
    strict_mask = fold_of[val_b] == fold
    return PairSplit(
        train_i=torch.as_tensor(
            dataset.pair_i[train_mask], dtype=torch.long, device=device
        ),
        train_j=torch.as_tensor(
            dataset.pair_j[train_mask], dtype=torch.long, device=device
        ),
        train_y=torch.as_tensor(
            dataset.pair_y[train_mask], dtype=torch.float32, device=device
        ),
        val_i=torch.as_tensor(val_a, dtype=torch.long, device=device),
        val_j=torch.as_tensor(val_b, dtype=torch.long, device=device),
        val_y=torch.as_tensor(val_y, dtype=torch.float32, device=device),
        strict_val_mask=strict_mask,
    )


def build_inner_split(
    dataset: Dataset,
    fold_of: np.ndarray,
    outer_fold: int,
    inner_fold: int,
    device: torch.device,
) -> PairSplit:
    """Use only the outer training molecules for nested tuning."""
    train_mask = (
        (fold_of[dataset.pair_i] != outer_fold)
        & (fold_of[dataset.pair_j] != outer_fold)
        & (fold_of[dataset.pair_i] != inner_fold)
        & (fold_of[dataset.pair_j] != inner_fold)
    )
    inner_a = fold_of[dataset.pair_i] == inner_fold
    inner_b = fold_of[dataset.pair_j] == inner_fold
    val_mask = ~train_mask & ~(
        (fold_of[dataset.pair_i] == outer_fold)
        | (fold_of[dataset.pair_j] == outer_fold)
    ) & (inner_a | inner_b)

    val_i_raw = dataset.pair_i[val_mask]
    val_j_raw = dataset.pair_j[val_mask]
    val_y_raw = dataset.pair_y[val_mask]
    target_a = inner_a[val_mask] & ~inner_b[val_mask]
    target_b = inner_b[val_mask] & ~inner_a[val_mask]
    val_a = np.where(target_a, val_i_raw, np.where(target_b, val_j_raw, val_i_raw))
    val_b = np.where(target_a, val_j_raw, np.where(target_b, val_i_raw, val_j_raw))
    val_y = np.where(
        target_a,
        val_y_raw,
        np.where(target_b, 1.0 - val_y_raw, val_y_raw),
    )
    strict_mask = inner_a[val_mask] & inner_b[val_mask]
    return PairSplit(
        train_i=torch.as_tensor(
            dataset.pair_i[train_mask], dtype=torch.long, device=device
        ),
        train_j=torch.as_tensor(
            dataset.pair_j[train_mask], dtype=torch.long, device=device
        ),
        train_y=torch.as_tensor(
            dataset.pair_y[train_mask], dtype=torch.float32, device=device
        ),
        val_i=torch.as_tensor(val_a, dtype=torch.long, device=device),
        val_j=torch.as_tensor(val_b, dtype=torch.long, device=device),
        val_y=torch.as_tensor(val_y, dtype=torch.float32, device=device),
        strict_val_mask=strict_mask,
    )


def model_from_config(
    kind: str,
    config: dict[str, Any],
    n_elements: int,
) -> torch.nn.Module:
    if kind == "wo":
        return WholeOnlyComparator(
            d_h=int(config["d_h"]),
            depth=int(config["depth"]),
            out=64,
            head_hid=int(config["head_hid"]),
            dropout=float(config["dropout"]),
        )
    if kind == "gt":
        return GraphormerComparator(
            n_elements,
            d=int(config["d"]),
            layers=int(config["layers"]),
            heads=int(config["heads"]),
            out=64,
            head_hid=int(config["head_hid"]),
            dropout=float(config["dropout"]),
        )
    raise ValueError(f"未知模型: {kind}")


def encode_all(
    kind: str,
    model: torch.nn.Module,
    dataset: Dataset,
) -> torch.Tensor:
    if kind == "wo":
        return model.encoder(dataset.wo_graphs)
    return model.embed(dataset.gt_batch)


def predict_pairs(
    kind: str,
    model: torch.nn.Module,
    dataset: Dataset,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    batch_size: int = 8192,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        embeddings = encode_all(kind, model, dataset)
        for start in range(0, len(pair_i), batch_size):
            end = min(start + batch_size, len(pair_i))
            probability = model.forward_emb(
                embeddings[pair_i[start:end]],
                embeddings[pair_j[start:end]],
            )
            outputs.append(probability.detach().cpu().numpy())
    if not outputs:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(outputs)


def classification_metrics(labels: np.ndarray, probability: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=int)
    probability = np.asarray(probability, dtype=float)
    result: dict[str, float | int] = {
        "pairs": int(len(labels)),
        "positive_pairs": int(labels.sum()),
    }
    if labels.size == 0:
        return result
    result["accuracy_at_0_5"] = float(accuracy_score(labels, probability >= 0.5))
    result["brier"] = float(brier_score_loss(labels, probability))
    if len(np.unique(labels)) == 2:
        result["auc"] = float(roc_auc_score(labels, probability))
    else:
        result["auc"] = float("nan")
    return result


def train_model(
    kind: str,
    config: dict[str, Any],
    dataset: Dataset,
    split: PairSplit,
    epochs: int,
    seed: int,
    device: torch.device,
    eval_interval: int,
    deterministic: bool,
    trial: optuna.Trial | None = None,
    log_prefix: str = "",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    seed_everything(seed, deterministic)
    model = model_from_config(kind, config, dataset.n_elements).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    batch_size = int(config["batch_size"])
    sample_count = len(split.train_i)
    if sample_count == 0:
        raise ValueError("训练配对为空，无法训练模型")
    history: list[dict[str, Any]] = []
    best_auc = -float("inf")
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(sample_count, device=device)
        total_loss = 0.0
        for start in range(0, sample_count, batch_size):
            batch_index = permutation[start:start + batch_size]
            embeddings = encode_all(kind, model, dataset)
            probability = model.forward_emb(
                embeddings[split.train_i[batch_index]],
                embeddings[split.train_j[batch_index]],
            )
            loss = F.binary_cross_entropy(
                probability, split.train_y[batch_index]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_index)

        completed_epoch = epoch + 1
        should_evaluate = (
            completed_epoch == 1
            or completed_epoch % eval_interval == 0
            or completed_epoch == epochs
        )
        if not should_evaluate:
            continue
        validation_probability = predict_pairs(
            kind,
            model,
            dataset,
            split.val_i,
            split.val_j,
        )
        validation_labels = split.val_y.detach().cpu().numpy()
        metrics = classification_metrics(
            validation_labels, validation_probability
        )
        auc = float(metrics["auc"])
        record = {
            "epoch": completed_epoch,
            "train_loss": total_loss / max(sample_count, 1),
            "val_auc": auc,
            "val_accuracy_at_0_5": metrics["accuracy_at_0_5"],
            "val_brier": metrics["brier"],
        }
        history.append(record)
        if np.isfinite(auc) and auc > best_auc:
            best_auc = auc
            best_epoch = completed_epoch
        logging.info(
            "%s epoch=%d loss=%.4f val_auc=%s val_acc=%.4f",
            log_prefix,
            completed_epoch,
            record["train_loss"],
            f"{auc:.4f}" if np.isfinite(auc) else "nan",
            float(metrics["accuracy_at_0_5"]),
        )
        if trial is not None:
            trial.report(auc, completed_epoch)
            if trial.should_prune():
                raise TrialPruned()

    return model, {
        "history": history,
        "best_val_auc": best_auc,
        "best_epoch_seen": best_epoch,
        "config": config,
        "seed": seed,
        "epochs": epochs,
    }


def suggest_config(kind: str, trial: optuna.Trial) -> dict[str, Any]:
    common = {
        "lr": trial.suggest_float("lr", 2e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float(
            "weight_decay", 1e-5, 1e-2, log=True
        ),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3, step=0.05),
    }
    if kind == "wo":
        return {
            **common,
            "d_h": trial.suggest_categorical("d_h", [128, 200, 256]),
            "depth": trial.suggest_int("depth", 2, 4),
            "head_hid": trial.suggest_categorical("head_hid", [32, 64, 96]),
            "batch_size": trial.suggest_categorical(
                "batch_size", [2048, 4096, 8192]
            ),
        }
    return {
        **common,
        "d": trial.suggest_categorical("d", [64, 96, 128]),
        "layers": trial.suggest_int("layers", 1, 3),
        "heads": trial.suggest_categorical("heads", [2, 4]),
        "head_hid": trial.suggest_categorical("head_hid", [32, 64]),
        "batch_size": trial.suggest_categorical(
            "batch_size", [1024, 2048, 4096]
        ),
    }


def config_signature(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def tune_for_outer_fold(
    kind: str,
    outer_fold: int,
    dataset: Dataset,
    fold_of: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
    outdir: Path,
) -> dict[str, Any]:
    inner_fold = (outer_fold + 1) % args.folds
    inner_split = build_inner_split(
        dataset, fold_of, outer_fold, inner_fold, device
    )
    study_dir = outdir / "studies"
    study_dir.mkdir(parents=True, exist_ok=True)
    storage_path = outdir / "optuna.sqlite3"
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{(storage_path.resolve()).as_posix()}",
        engine_kwargs={"connect_args": {"timeout": 30}},
    )
    study = optuna.create_study(
        study_name=f"{kind}_outer{outer_fold}",
        direction="maximize",
        sampler=TPESampler(seed=args.tuning_seed + outer_fold),
        pruner=HyperbandPruner(
            min_resource=max(1, args.eval_interval),
            max_resource=args.epochs,
            reduction_factor=3,
        ),
        storage=storage,
        load_if_exists=args.resume,
    )

    def objective(trial: optuna.Trial) -> float:
        config = suggest_config(kind, trial)
        logging.info(
            "TUNE model=%s outer=%d inner=%d trial=%d config=%s",
            kind,
            outer_fold,
            inner_fold,
            trial.number,
            json.dumps(config, sort_keys=True),
        )
        try:
            _, result = train_model(
                kind,
                config,
                dataset,
                inner_split,
                args.epochs,
                args.tuning_seed + outer_fold * 100 + trial.number,
                device,
                args.eval_interval,
                args.deterministic,
                trial=trial,
                log_prefix=f"tune/{kind}/outer{outer_fold}/trial{trial.number}",
            )
            trial.set_user_attr("history", result["history"])
            return float(result["best_val_auc"])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logging.exception("Tuning trial ran out of GPU memory")
            raise
        finally:
            release_accelerator_memory(device)

    completed_before = len(study.trials)
    remaining = max(0, args.trials_per_fold - completed_before)
    if remaining:
        study.optimize(objective, n_trials=remaining)
    completed = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        raise RuntimeError(f"{kind} outer fold {outer_fold} 没有完成任何调参试验")
    best = study.best_trial
    result = {
        "model": kind,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "inner_train_pairs": int(len(inner_split.train_i)),
        "inner_val_pairs": int(len(inner_split.val_i)),
        "best_value": float(best.value),
        "best_trial": int(best.number),
        "best_config": dict(best.params),
        "all_trials": [
            {
                "number": int(trial.number),
                "state": str(trial.state),
                "value": None if trial.value is None else float(trial.value),
                "params": dict(trial.params),
                "duration_seconds": (
                    float(trial.duration.total_seconds())
                    if trial.duration is not None
                    else None
                ),
            }
            for trial in study.trials
        ],
    }
    path = study_dir / f"{kind}_outer{outer_fold}.json"
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logging.info(
        "TUNE_DONE model=%s outer=%d best_trial=%d auc=%.4f",
        kind,
        outer_fold,
        best.number,
        float(best.value),
    )
    return result


def default_config(kind: str) -> dict[str, Any]:
    if kind == "wo":
        return {
            "lr": 1e-3,
            "weight_decay": 3e-3,
            "dropout": 0.1,
            "d_h": 200,
            "depth": 3,
            "head_hid": 64,
            "batch_size": 4096,
        }
    return {
        "lr": 1e-3,
        "weight_decay": 3e-3,
        "dropout": 0.1,
        "d": 128,
        "layers": 2,
        "heads": 4,
        "head_hid": 64,
        "batch_size": 2048,
    }


def checkpoint_name(kind: str, epochs: int, seed: int) -> str:
    return f"full292_{MODEL_LABELS[kind]}_x{epochs}_s{seed}.pt"


def save_state(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    torch.save(state, path)


def train_fold_models(
    kind: str,
    fold: int,
    config: dict[str, Any],
    dataset: Dataset,
    split: PairSplit,
    device: torch.device,
    args: argparse.Namespace,
    outdir: Path,
) -> list[dict[str, Any]]:
    model_dir = outdir / "checkpoints" / f"fold_{fold}"
    model_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for seed in args.seeds:
        path = model_dir / checkpoint_name(kind, args.epochs, seed)
        history_path = model_dir / f"{kind}_s{seed}_history.json"
        prediction_path = model_dir / f"{kind}_s{seed}_validation.npy"
        if args.resume and path.exists() and history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
            results.append(
                {
                    "model": kind,
                    "fold": fold,
                    "seed": seed,
                    "checkpoint": str(path),
                    "resumed": True,
                    "history": history,
                }
            )
            continue
        model, result = train_model(
            kind,
            config,
            dataset,
            split,
            args.epochs,
            seed + (1000 if kind == "gt" else 0),
            device,
            args.eval_interval,
            args.deterministic,
            log_prefix=f"final/{kind}/fold{fold}/seed{seed}",
        )
        save_state(model, path)
        history_path.write_text(
            json.dumps(result["history"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        validation_probability = predict_pairs(
            kind, model, dataset, split.val_i, split.val_j
        )
        np.save(prediction_path, validation_probability)
        results.append(
            {
                "model": kind,
                "fold": fold,
                "seed": seed,
                "checkpoint": str(path),
                "resumed": False,
                **result,
            }
        )
        del model
        release_accelerator_memory(
            next(iter(dataset.gt_batch)).device
        )
    return results


def evaluate_fold(
    fold: int,
    dataset: Dataset,
    split: PairSplit,
    fold_configs: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    outdir: Path,
) -> dict[str, Any]:
    model_dir = outdir / "checkpoints" / f"fold_{fold}"
    component_probabilities: dict[str, list[np.ndarray]] = {
        kind: [] for kind in args.models
    }
    for kind in args.models:
        for seed in args.seeds:
            path = model_dir / checkpoint_name(kind, args.epochs, seed)
            if not path.exists():
                raise FileNotFoundError(f"缺少折内模型: {path}")
            model = model_from_config(
                kind, fold_configs[kind], dataset.n_elements
            ).to(next(iter(dataset.gt_batch)).device)
            state = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            component_probabilities[kind].append(
                predict_pairs(kind, model, dataset, split.val_i, split.val_j)
            )
            del model

    component_mean = {
        kind: np.mean(np.stack(values), axis=0)
        for kind, values in component_probabilities.items()
    }
    ensemble_probability = np.mean(
        np.stack([component_mean[kind] for kind in args.models]), axis=0
    )
    labels = split.val_y.detach().cpu().numpy().astype(int)
    strict_mask = split.strict_val_mask
    val_i = split.val_i.detach().cpu().numpy()
    val_j = split.val_j.detach().cpu().numpy()
    output = pd.DataFrame(
        {
            "fold": fold,
            "heldout_target_smiles": [dataset.smiles[i] for i in val_i],
            "partner_smiles": [dataset.smiles[j] for j in val_j],
            "label_target_better": labels,
            "p_target_better": ensemble_probability,
            "strict_both_molecules_unseen": strict_mask,
        }
    )
    for kind, probability in component_mean.items():
        output[f"p_{kind}"] = probability
    return {
        "fold": fold,
        "train_pairs": int(len(split.train_i)),
        "val_pairs": int(len(split.val_i)),
        "strict_unseen_pairs": int(strict_mask.sum()),
        "cold_start_metrics": classification_metrics(labels, ensemble_probability),
        "strict_unseen_metrics": classification_metrics(
            labels[strict_mask], ensemble_probability[strict_mask]
        ),
        "component_cold_start_metrics": {
            kind: classification_metrics(labels, probability)
            for kind, probability in component_mean.items()
        },
        "configs": fold_configs,
        "predictions": output,
    }


def choose_final_config(
    kind: str,
    fold_results: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    candidates: list[tuple[float, dict[str, Any], int]] = []
    for fold, result in fold_results.items():
        value = float(result["best_value"])
        candidates.append((value, result["best_config"], fold))
    candidates.sort(key=lambda item: item[0], reverse=True)
    config = candidates[0][1]
    logging.info(
        "FINAL_CONFIG model=%s fold=%d inner_auc=%.4f config=%s",
        kind,
        candidates[0][2],
        candidates[0][0],
        json.dumps(config, sort_keys=True),
    )
    return config


def train_full_models(
    kind: str,
    config: dict[str, Any],
    dataset: Dataset,
    device: torch.device,
    args: argparse.Namespace,
    outdir: Path,
) -> None:
    training_split = PairSplit(
        train_i=torch.as_tensor(dataset.pair_i, dtype=torch.long, device=device),
        train_j=torch.as_tensor(dataset.pair_j, dtype=torch.long, device=device),
        train_y=torch.as_tensor(
            dataset.pair_y, dtype=torch.float32, device=device
        ),
        val_i=torch.as_tensor(dataset.pair_i, dtype=torch.long, device=device),
        val_j=torch.as_tensor(dataset.pair_j, dtype=torch.long, device=device),
        val_y=torch.as_tensor(
            dataset.pair_y, dtype=torch.float32, device=device
        ),
        strict_val_mask=np.ones(len(dataset.pair_i), dtype=bool),
    )
    final_dir = outdir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        path = final_dir / checkpoint_name(kind, args.epochs, seed)
        history_path = final_dir / f"{kind}_s{seed}_history.json"
        if args.resume and path.exists() and history_path.exists():
            logging.info("SKIP existing final checkpoint %s", path.name)
            continue
        model, result = train_model(
            kind,
            config,
            dataset,
            training_split,
            args.epochs,
            seed + (1000 if kind == "gt" else 0),
            device,
            args.eval_interval,
            args.deterministic,
            log_prefix=f"full/{kind}/seed{seed}",
        )
        save_state(model, path)
        history_path.write_text(
            json.dumps(result["history"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        del model
        release_accelerator_memory(
            next(iter(dataset.gt_batch)).device
        )


def write_split_manifest(
    outdir: Path,
    dataset: Dataset,
    fold_of: np.ndarray,
    pair_fold: np.ndarray,
    target_side: np.ndarray,
) -> None:
    molecule_folds = {
        str(fold): [
            dataset.smiles[index]
            for index in np.flatnonzero(fold_of == fold)
        ]
        for fold in range(int(fold_of.max()) + 1)
    }
    pair_folds = {
        str(fold): int((pair_fold == fold).sum())
        for fold in range(int(fold_of.max()) + 1)
    }
    manifest = {
        "policy": "cold_start_target_molecule",
        "notes": (
            "Each unordered pair is evaluated once. The queried target "
            "molecule belongs to the held-out fold and never appears in that "
            "fold's training set. The partner may be a known training "
            "molecule. The strict subset where both endpoint molecules are "
            "held out is reported separately."
        ),
        "unordered_pairs": int(len(dataset.pair_i)),
        "molecule_folds": molecule_folds,
        "pair_fold_counts": pair_folds,
        "pair_fold": pair_fold.tolist(),
        "target_side": target_side.tolist(),
    }
    (outdir / "splits.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs 必须为正整数")
    if args.folds < 3:
        raise ValueError("--folds 至少为 3，嵌套调参需要独立的训练/验证折")
    if args.trials_per_fold <= 0:
        raise ValueError("--trials-per-fold 必须为正整数")
    if args.eval_interval <= 0:
        raise ValueError("--eval-interval 必须为正整数")
    if not args.seeds:
        raise ValueError("--seeds 不能为空")

    outdir = args.outdir.expanduser().resolve()
    if outdir.exists() and not args.resume:
        if any(outdir.iterdir()):
            raise FileExistsError(
                f"输出目录非空且未指定 --resume: {outdir}"
            )
    configure_logging(outdir)
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        logging.info(
            "DEVICE cuda name=%s capability=%s memory_gb=%.2f",
            torch.cuda.get_device_name(0),
            ".".join(map(str, torch.cuda.get_device_capability(0))),
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
        )
    else:
        logging.info("DEVICE cpu")

    dataset = load_dataset()
    to_device_dataset(dataset, device)
    fold_of, pair_fold, target_side = assign_molecule_folds(
        dataset, args.folds, args.split_seed
    )
    write_split_manifest(
        outdir, dataset, fold_of, pair_fold, target_side
    )
    logging.info(
        "DATA molecules=%d unordered_pairs=%d fold_molecule_counts=%s "
        "fold_val_pair_counts=%s",
        len(dataset.smiles),
        len(dataset.pair_i),
        np.bincount(fold_of).tolist(),
        np.bincount(pair_fold).tolist(),
    )

    config = {
        "epochs": args.epochs,
        "folds": args.folds,
        "seeds": args.seeds,
        "models": args.models,
        "trials_per_fold": args.trials_per_fold,
        "eval_interval": args.eval_interval,
        "split_seed": args.split_seed,
        "tuning_seed": args.tuning_seed,
        "deterministic": args.deterministic,
        "device": str(device),
        "torch": torch.__version__,
        "optuna": optuna.__version__,
    }
    (outdir / "experiment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    tuning_results: dict[str, dict[int, dict[str, Any]]] = {
        kind: {} for kind in args.models
    }
    for kind in args.models:
        for fold in range(args.folds):
            summary_path = outdir / "studies" / f"{kind}_outer{fold}.json"
            if args.skip_tuning:
                tuning_results[kind][fold] = {
                    "best_value": float("nan"),
                    "best_trial": -1,
                    "best_config": default_config(kind),
                }
            elif args.resume and summary_path.exists():
                tuning_results[kind][fold] = json.loads(
                    summary_path.read_text(encoding="utf-8")
                )
                logging.info(
                    "SKIP tuning model=%s outer=%d", kind, fold
                )
            else:
                tuning_results[kind][fold] = tune_for_outer_fold(
                    kind,
                    fold,
                    dataset,
                    fold_of,
                    device,
                    args,
                    outdir,
                )
    if args.tune_only:
        logging.info("TUNE_ONLY complete")
        return

    fold_metrics: dict[str, Any] = {}
    all_predictions: list[pd.DataFrame] = []
    for fold in range(args.folds):
        split = build_outer_split(
            dataset, fold_of, pair_fold, target_side, fold, device
        )
        fold_configs = {
            kind: tuning_results[kind][fold]["best_config"]
            for kind in args.models
        }
        logging.info(
            "OUTER_FOLD %d train=%d val=%d strict_subset=%d",
            fold,
            len(split.train_i),
            len(split.val_i),
            int(split.strict_val_mask.sum()),
        )
        for kind in args.models:
            train_fold_models(
                kind,
                fold,
                fold_configs[kind],
                dataset,
                split,
                device,
                args,
                outdir,
            )
        fold_result = evaluate_fold(
            fold, dataset, split, fold_configs, args, outdir
        )
        predictions = fold_result.pop("predictions")
        all_predictions.append(predictions)
        fold_metrics[str(fold)] = fold_result
        metric_path = outdir / "metrics" / f"fold_{fold}.json"
        metric_path.parent.mkdir(parents=True, exist_ok=True)
        metric_path.write_text(
            json.dumps(fold_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logging.info(
            "FOLD_DONE %d cold_auc=%.4f strict_auc=%s",
            fold,
            float(fold_result["cold_start_metrics"]["auc"]),
            fold_result["strict_unseen_metrics"].get("auc"),
        )
        if device.type == "cuda":
            logging.info(
                "GPU_MEMORY allocated_peak_gb=%.2f reserved_peak_gb=%.2f",
                torch.cuda.max_memory_allocated() / 1024**3,
                torch.cuda.max_memory_reserved() / 1024**3,
            )
            torch.cuda.reset_peak_memory_stats()

    oof = pd.concat(all_predictions, ignore_index=True)
    oof_path = outdir / "metrics" / "oof_predictions.csv"
    oof.to_csv(oof_path, index=False, encoding="utf-8-sig")
    labels = oof["label_target_better"].to_numpy(dtype=int)
    probability = oof["p_target_better"].to_numpy(dtype=float)
    strict_mask = oof["strict_both_molecules_unseen"].to_numpy(dtype=bool)
    fold_aucs = [
        float(value["cold_start_metrics"]["auc"])
        for value in fold_metrics.values()
    ]
    summary = {
        "cv": fold_metrics,
        "pooled_oof_cold_start": classification_metrics(labels, probability),
        "pooled_oof_strict_unseen": classification_metrics(
            labels[strict_mask], probability[strict_mask]
        ),
        "fold_cold_start_auc_mean": float(np.mean(fold_aucs)),
        "fold_cold_start_auc_std": float(np.std(fold_aucs)),
    }
    (outdir / "metrics" / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logging.info(
        "CV_DONE pooled_cold_auc=%.4f pooled_strict_auc=%s",
        float(summary["pooled_oof_cold_start"]["auc"]),
        summary["pooled_oof_strict_unseen"].get("auc"),
    )

    if not args.skip_final_full:
        final_config = {
            kind: choose_final_config(kind, tuning_results[kind])
            for kind in args.models
        }
        (outdir / "final" / "ensemble_config.json").parent.mkdir(
            parents=True, exist_ok=True
        )
        (outdir / "final" / "ensemble_config.json").write_text(
            json.dumps(final_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        for kind in args.models:
            train_full_models(
                kind,
                final_config[kind],
                dataset,
                device,
                args,
                outdir,
            )
    logging.info("ALL_DONE output=%s", outdir)


if __name__ == "__main__":
    main()
