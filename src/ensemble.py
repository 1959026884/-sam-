# -*- coding: utf-8 -*-
"""Checkpoint loading and efficient six-model ensemble inference."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

try:
    from .common import (
        DEFAULT_EPOCHS,
        ELEMENT_TO_INDEX,
        SEEDS,
        checkpoint_path,
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
        DEFAULT_EPOCHS,
        ELEMENT_TO_INDEX,
        SEEDS,
        checkpoint_path,
    )
    from models import (
        GraphormerComparator,
        WholeOnlyComparator,
        build_batch,
        graph_from_mol,
        molecule_from_smiles,
    )


def _load_state_dict(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"缺少检查点: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


_WO_ARCHITECTURE_KEYS = {"d_h", "depth", "out", "head_hid", "dropout"}
_GT_ARCHITECTURE_KEYS = {
    "d",
    "layers",
    "heads",
    "out",
    "head_hid",
    "dropout",
}


def _architecture_config(kind: str, config: dict) -> dict:
    allowed = (
        _WO_ARCHITECTURE_KEYS if kind == "wo" else _GT_ARCHITECTURE_KEYS
    )
    return {
        key: value for key, value in config.items() if key in allowed
    }


class ModelEnsemble:
    """Three D-MPNN seeds plus three Graphormer seeds."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        epochs: int = DEFAULT_EPOCHS,
        seeds: Sequence[int] = SEEDS,
    ):
        self.checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        self.epochs = epochs
        self.seeds = tuple(seeds)
        self.wo_models: list[WholeOnlyComparator] = []
        self.gt_models: list[GraphormerComparator] = []

        config_path = self.checkpoint_dir / "ensemble_config.json"
        model_configs = {}
        if config_path.is_file():
            model_configs = json.loads(config_path.read_text(encoding="utf-8"))
        wo_config = _architecture_config("wo", model_configs.get("wo", {}))
        gt_config = _architecture_config("gt", model_configs.get("gt", {}))
        n_elements = len(ELEMENT_TO_INDEX) + 2
        for seed in self.seeds:
            wo_path = checkpoint_path(
                self.checkpoint_dir, "wo", epochs, seed
            )
            wo_model = WholeOnlyComparator(**wo_config)
            wo_model.load_state_dict(_load_state_dict(wo_path))
            wo_model.eval()
            self.wo_models.append(wo_model)

            gt_path = checkpoint_path(
                self.checkpoint_dir, "gt", epochs, seed
            )
            gt_model = GraphormerComparator(n_elements, **gt_config)
            gt_model.load_state_dict(_load_state_dict(gt_path))
            gt_model.eval()
            self.gt_models.append(gt_model)

    @staticmethod
    def _normalize_smiles(values: Sequence[str]) -> list[str]:
        return [str(value).strip() for value in values]

    def _encode(
        self,
        unique_smiles: Sequence[str],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if not unique_smiles:
            return [], []

        mols = [molecule_from_smiles(smiles) for smiles in unique_smiles]
        graphs = [graph_from_mol(mol) for mol in mols]
        gt_batch = build_batch(mols, ELEMENT_TO_INDEX)

        with torch.no_grad():
            wo_embeddings = [
                model.embed_all(graphs) for model in self.wo_models
            ]
            gt_embeddings = [
                model.embed(gt_batch) for model in self.gt_models
            ]
        return wo_embeddings, gt_embeddings

    def predict_pairs(
        self,
        smiles_a: Sequence[str],
        smiles_b: Sequence[str],
        *,
        batch_size: int = 8192,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ensemble, D-MPNN, and Graphormer P(A better than B)."""
        smiles_a = self._normalize_smiles(smiles_a)
        smiles_b = self._normalize_smiles(smiles_b)
        if len(smiles_a) != len(smiles_b):
            raise ValueError("smiles_a 与 smiles_b 数量不一致")
        if not smiles_a:
            empty = np.empty(0, dtype=np.float32)
            return empty, empty.copy(), empty.copy()

        unique_smiles = list(dict.fromkeys(smiles_a + smiles_b))
        lookup = {smiles: index for index, smiles in enumerate(unique_smiles)}
        wo_embeddings, gt_embeddings = self._encode(unique_smiles)

        wo_outputs = []
        gt_outputs = []
        with torch.no_grad():
            for start in range(0, len(smiles_a), batch_size):
                end = min(start + batch_size, len(smiles_a))
                index_a = torch.tensor(
                    [lookup[value] for value in smiles_a[start:end]],
                    dtype=torch.long,
                )
                index_b = torch.tensor(
                    [lookup[value] for value in smiles_b[start:end]],
                    dtype=torch.long,
                )

                wo_seed_predictions = [
                    model.forward_emb(
                        embedding[index_a], embedding[index_b]
                    )
                    for model, embedding in zip(
                        self.wo_models, wo_embeddings
                    )
                ]
                gt_seed_predictions = [
                    model.forward_emb(
                        embedding[index_a], embedding[index_b]
                    )
                    for model, embedding in zip(
                        self.gt_models, gt_embeddings
                    )
                ]
                wo_outputs.append(torch.stack(wo_seed_predictions).mean(0))
                gt_outputs.append(torch.stack(gt_seed_predictions).mean(0))

        wo_probability = torch.cat(wo_outputs).cpu().numpy()
        gt_probability = torch.cat(gt_outputs).cpu().numpy()
        ensemble_probability = (wo_probability + gt_probability) / 2.0
        return ensemble_probability, wo_probability, gt_probability
