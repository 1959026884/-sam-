# -*- coding: utf-8 -*-
"""Model definitions shared by training, prediction, and evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from chemprop.data.collate import BatchMolGraph
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.featurizers.atom import MultiHotAtomFeaturizer
from chemprop.featurizers.bond import MultiHotBondFeaturizer
from chemprop.nn import BondMessagePassing, MeanAggregation
from rdkit import Chem

try:
    from .common import ELEMENT_TO_INDEX, SUPPORTED_ELEMENTS
except ImportError:
    from common import ELEMENT_TO_INDEX, SUPPORTED_ELEMENTS


DMAX = 32
ATOM_CAP = 112
BOND_IDX = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}

_FEATURIZER = SimpleMoleculeMolGraphFeaturizer(
    atom_featurizer=MultiHotAtomFeaturizer.v2(),
    bond_featurizer=MultiHotBondFeaturizer(),
)


def molecule_from_smiles(smiles: str, *, atom_cap: int = ATOM_CAP) -> Chem.Mol:
    """Parse and validate a SMILES before it reaches either model."""
    text = str(smiles).strip()
    if not text or text.lower() == "nan":
        raise ValueError("SMILES 为空")

    mol = Chem.MolFromSmiles(text)
    if mol is None:
        raise ValueError(f"SMILES 解析失败: {text}")
    if mol.GetNumAtoms() == 0:
        raise ValueError(f"SMILES 不含原子: {text}")
    if mol.GetNumAtoms() > atom_cap:
        raise ValueError(
            f"分子含 {mol.GetNumAtoms()} 个重原子，超过 Graphormer 上限 "
            f"{atom_cap}: {text}"
        )

    unsupported = sorted(
        {atom.GetSymbol() for atom in mol.GetAtoms()} - SUPPORTED_ELEMENTS
    )
    if unsupported:
        raise ValueError(f"模型不支持元素 {', '.join(unsupported)}: {text}")
    return mol


def graph_from_mol(mol: Chem.Mol):
    return _FEATURIZER(mol)


def graph_of(smiles: str):
    return graph_from_mol(molecule_from_smiles(smiles))


def bond_feat(mol: Chem.Mol):
    n = mol.GetNumAtoms()
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = BOND_IDX.get(str(bond.GetBondType()), 0)
        adjacency[i].append((j, bond_type))
        adjacency[j].append((i, bond_type))

    distance = np.full((n, n), -1, dtype=np.int64)
    bond_counts = np.zeros((n, n, 4), dtype=np.float32)
    for start in range(n):
        distance[start, start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            current = queue[head]
            head += 1
            for neighbor, bond_type in adjacency[current]:
                if distance[start, neighbor] == -1:
                    distance[start, neighbor] = distance[start, current] + 1
                    bond_counts[start, neighbor] = bond_counts[start, current]
                    bond_counts[start, neighbor, bond_type] += 1.0
                    queue.append(neighbor)
    return distance, bond_counts


def mol_feats(mol: Chem.Mol, element_index: dict[str, int]):
    n = mol.GetNumAtoms()
    elements = np.zeros(n, dtype=np.int64)
    scalar = np.zeros((n, 4), dtype=np.float32)
    for atom in mol.GetAtoms():
        index = atom.GetIdx()
        elements[index] = element_index.get(atom.GetSymbol(), 1)
        scalar[index] = [
            atom.GetDegree(),
            atom.GetTotalNumHs(),
            float(atom.GetIsAromatic()),
            atom.GetFormalCharge(),
        ]
    distance, bond_counts = bond_feat(mol)
    return elements, scalar, distance, bond_counts


def bucketize(distance: np.ndarray) -> np.ndarray:
    buckets = np.zeros_like(distance, dtype=np.int64)
    buckets[distance == 0] = 1
    nearby = (distance >= 1) & (distance <= DMAX)
    buckets[nearby] = distance[nearby] + 1
    buckets[~((distance == 0) | nearby)] = DMAX + 2
    return buckets


def build_batch(
    mols: Sequence[Chem.Mol],
    element_index: dict[str, int] | None = None,
):
    if not mols:
        raise ValueError("不能为空分子列表构造批数据")
    for mol in mols:
        if mol is None:
            raise ValueError("分子列表中包含解析失败的分子")
        if mol.GetNumAtoms() > ATOM_CAP:
            raise ValueError(
                f"分子含 {mol.GetNumAtoms()} 个原子，超过上限 {ATOM_CAP}"
            )

    element_index = element_index or ELEMENT_TO_INDEX
    atom_dim = max(mol.GetNumAtoms() for mol in mols)
    batch_size = len(mols)
    elements = np.zeros((batch_size, atom_dim), dtype=np.int64)
    scalar = np.zeros((batch_size, atom_dim, 4), dtype=np.float32)
    distance_buckets = np.full(
        (batch_size, atom_dim, atom_dim), DMAX + 2, dtype=np.int64
    )
    bond_encoding = np.zeros(
        (batch_size, atom_dim, atom_dim, 4), dtype=np.float32
    )
    distance_values = np.zeros(
        (batch_size, atom_dim, atom_dim), dtype=np.float32
    )
    mask = np.zeros((batch_size, atom_dim), dtype=bool)

    for batch_index, mol in enumerate(mols):
        atom_elements, atom_scalar, distance, bond_counts = mol_feats(
            mol, element_index
        )
        n = len(atom_elements)
        elements[batch_index, :n] = atom_elements
        scalar[batch_index, :n] = atom_scalar
        distance_buckets[batch_index, :n, :n] = bucketize(distance[:n, :n])
        bond_encoding[batch_index, :n, :n] = bond_counts[:n, :n]
        distance_values[batch_index, :n, :n] = distance[:n, :n]
        mask[batch_index, :n] = True

    return (
        torch.tensor(elements),
        torch.tensor(scalar),
        torch.tensor(distance_buckets),
        torch.tensor(bond_encoding),
        torch.tensor(distance_values),
        torch.tensor(mask),
    )


class FragGraphEncoder(nn.Module):
    def __init__(
        self,
        d_h: int = 200,
        depth: int = 3,
        out: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mpnn = BondMessagePassing(d_h=d_h, depth=depth, dropout=dropout)
        self.agg = MeanAggregation()
        self.proj = nn.Sequential(
            nn.Linear(d_h, out),
            nn.LayerNorm(out),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(out, out),
        )

    def forward(self, graph_batch):
        hidden = self.mpnn(graph_batch)
        embedding = self.agg(hidden, graph_batch.batch)
        return self.proj(embedding)


class WholeOnlyComparator(nn.Module):
    """Whole-molecule D-MPNN encoder with an antisymmetric difference head."""

    def __init__(
        self,
        d_h: int = 200,
        depth: int = 3,
        out: int = 64,
        head_hid: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = FragGraphEncoder(d_h, depth, out, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(out, head_hid, bias=False),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(head_hid, head_hid // 2, bias=False),
            nn.Tanh(),
            nn.Linear(head_hid // 2, 1, bias=False),
        )

    def embed_all(self, graphs):
        graph_batch = BatchMolGraph(graphs)
        return self.encoder(graph_batch)

    def forward_emb(self, embedding_a, embedding_b):
        return torch.sigmoid(self.ffn(embedding_b - embedding_a).squeeze(-1))


class GraphormerLayer(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d),
        )
        self.heads = heads
        self.head_dim = d // heads
        self.drop = nn.Dropout(dropout)

    def forward(self, x, bias, mask):
        batch_size, atom_dim, _ = x.shape
        hidden = self.ln1(x)
        q = self.wq(hidden).view(
            batch_size, atom_dim, self.heads, self.head_dim
        ).transpose(1, 2)
        k = self.wk(hidden).view(
            batch_size, atom_dim, self.heads, self.head_dim
        ).transpose(1, 2)
        v = self.wv(hidden).view(
            batch_size, atom_dim, self.heads, self.head_dim
        ).transpose(1, 2)
        score = q @ k.transpose(-1, -2) / (self.head_dim ** 0.5)
        score = score + bias.unsqueeze(1)
        pair_mask = mask[:, None, :] & mask[:, :, None]
        score = score.masked_fill(~pair_mask.unsqueeze(1), -1e9)
        attention = F.softmax(score, dim=-1)
        output = (
            attention @ v
        ).transpose(1, 2).reshape(batch_size, atom_dim, -1)
        x = x + self.drop(self.out(output))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class GraphormerEncoder(nn.Module):
    def __init__(
        self,
        n_elements: int,
        d: int = 128,
        layers: int = 2,
        heads: int = 4,
        out: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.elem = nn.Embedding(n_elements, d, padding_idx=0)
        self.feat = nn.Linear(4, d, bias=False)
        self.sp = nn.Embedding(DMAX + 3, 1)
        self.edge_lin = nn.Linear(4, d, bias=False)
        self.edge_u = nn.Parameter(torch.zeros(d))
        self.layers = nn.ModuleList(
            [GraphormerLayer(d, heads, dropout) for _ in range(layers)]
        )
        self.proj = nn.Sequential(
            nn.Linear(d, out),
            nn.LayerNorm(out),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(out, out),
        )

    def bias_of(self, distance_buckets, bond_counts, distance_values):
        spatial_bias = self.sp(distance_buckets).squeeze(-1)
        edge_embedding = self.edge_lin(bond_counts)
        edge_bias = (
            edge_embedding / distance_values.unsqueeze(-1).clamp(min=1.0)
        ) @ self.edge_u
        return spatial_bias + edge_bias

    def forward(
        self,
        elements,
        scalar,
        distance_buckets,
        bond_counts,
        distance_values,
        mask,
    ):
        x = self.elem(elements) + self.feat(scalar)
        bias = self.bias_of(distance_buckets, bond_counts, distance_values)
        for layer in self.layers:
            x = layer(x, bias, mask)
        mask_float = mask.unsqueeze(-1).float()
        pooled = (x * mask_float).sum(1) / mask_float.sum(1).clamp(min=1.0)
        return self.proj(pooled)


class GraphormerComparator(nn.Module):
    def __init__(
        self,
        n_elements: int,
        d: int = 128,
        layers: int = 2,
        heads: int = 4,
        out: int = 64,
        head_hid: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = GraphormerEncoder(
            n_elements, d, layers, heads, out, dropout
        )
        self.ffn = nn.Sequential(
            nn.Linear(out, head_hid, bias=False),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(head_hid, head_hid // 2, bias=False),
            nn.Tanh(),
            nn.Linear(head_hid // 2, 1, bias=False),
        )

    def embed(self, batch):
        return self.encoder(*batch)

    def forward_emb(self, embedding_a, embedding_b):
        return torch.sigmoid(self.ffn(embedding_b - embedding_a).squeeze(-1))
