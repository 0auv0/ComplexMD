"""SE(3)-equivariant building blocks for joint ligand space-time modeling."""

from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=values.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = values.float().unsqueeze(-1) * frequencies
    embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
    if embedding.shape[-1] < dim:
        embedding = torch.nn.functional.pad(embedding, (0, dim - embedding.shape[-1]))
    return embedding


class RBF(nn.Module):
    def __init__(self, count: int, maximum: float):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, maximum, count))
        self.gamma = float(count) / maximum

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (distance.unsqueeze(-1) - self.centers) ** 2)


def _rotary_part(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    width = x.shape[-1]
    if width == 0:
        return x
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(0, width, 2, device=x.device, dtype=torch.float32)
        / max(width, 2)
    )
    angle = positions.float().view(1, 1, -1, 1) * frequencies
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack(
        [even * angle.cos() - odd * angle.sin(), even * angle.sin() + odd * angle.cos()],
        dim=-1,
    ).flatten(-2)


def apply_2d_rope(
    tensor: torch.Tensor, time_ids: torch.Tensor, atom_ids: torch.Tensor
) -> torch.Tensor:
    """Apply independent rotary phases for frame and atom axes."""
    width = tensor.shape[-1]
    time_width = (width // 4) * 2
    atom_width = ((width - time_width) // 2) * 2
    tail_start = time_width + atom_width
    return torch.cat(
        [
            _rotary_part(tensor[..., :time_width], time_ids),
            _rotary_part(tensor[..., time_width:tail_start], atom_ids),
            tensor[..., tail_start:],
        ],
        dim=-1,
    )


class JointSpaceTimeAttention(nn.Module):
    """Full atom-frame attention with causal time masking and geometric bias."""

    def __init__(
        self, hidden_dim: int, num_heads: int, num_rbf: int, rbf_max: float, dropout: float
    ):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.rbf = RBF(num_rbf, rbf_max)
        self.distance_bias = nn.Linear(num_rbf, num_heads, bias=False)
        self.relative_time_bias = nn.Embedding(33, num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        coordinates: torch.Tensor,
        valid: torch.Tensor,
        time_ids: torch.Tensor,
        atom_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, hidden = tokens.shape
        qkv = self.qkv(tokens).view(batch, length, 3, self.num_heads, self.head_dim)
        q, k, value = [x.transpose(1, 2) for x in qkv.unbind(dim=2)]
        q = apply_2d_rope(q, time_ids, atom_ids)
        k = apply_2d_rope(k, time_ids, atom_ids)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        distance = torch.cdist(coordinates, coordinates)
        logits = logits + self.distance_bias(self.rbf(distance)).permute(0, 3, 1, 2)
        delta_t = (time_ids[:, None] - time_ids[None, :]).clamp(-16, 16) + 16
        logits = logits + self.relative_time_bias(delta_t).permute(2, 0, 1).unsqueeze(0)
        causal = time_ids[None, :] <= time_ids[:, None]
        allowed = causal.view(1, 1, length, length) & valid[:, None, None, :]
        logits = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
        attention = self.dropout(torch.softmax(logits.float(), dim=-1).to(logits.dtype))
        output = torch.matmul(attention, value).transpose(1, 2).reshape(batch, length, hidden)
        return self.output(output) * valid.unsqueeze(-1)


class PocketCrossAttention(nn.Module):
    def __init__(
        self, hidden_dim: int, num_heads: int, num_rbf: int, rbf_max: float, dropout: float
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.rbf = RBF(num_rbf, rbf_max)
        self.distance_bias = nn.Linear(num_rbf, num_heads, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        ligand: torch.Tensor,
        ligand_coordinates: torch.Tensor,
        ligand_valid: torch.Tensor,
        pocket: torch.Tensor,
        pocket_ca: torch.Tensor,
        pocket_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, hidden = ligand.shape
        pocket_length = pocket.shape[1]
        q = self.query(ligand).view(batch, length, self.num_heads, self.head_dim)
        k = self.key(pocket).view(batch, pocket_length, self.num_heads, self.head_dim)
        value = self.value(pocket).view(batch, pocket_length, self.num_heads, self.head_dim)
        q, k, value = [x.transpose(1, 2) for x in (q, k, value)]
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        distance = torch.cdist(ligand_coordinates, pocket_ca)
        logits = logits + self.distance_bias(self.rbf(distance)).permute(0, 3, 1, 2)
        logits = logits.masked_fill(
            ~pocket_valid[:, None, None, :], torch.finfo(logits.dtype).min
        )
        attention = self.dropout(torch.softmax(logits.float(), dim=-1).to(logits.dtype))
        output = torch.matmul(attention, value).transpose(1, 2).reshape(batch, length, hidden)
        return self.output(output) * ligand_valid.unsqueeze(-1)


class JointSpaceTimeBlock(nn.Module):
    def __init__(
        self, hidden_dim: int, num_heads: int, num_rbf: int, rbf_max: float, dropout: float
    ):
        super().__init__()
        self.norm_joint = nn.LayerNorm(hidden_dim)
        self.norm_cross = nn.LayerNorm(hidden_dim)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.joint = JointSpaceTimeAttention(
            hidden_dim, num_heads, num_rbf, rbf_max, dropout
        )
        self.cross = PocketCrossAttention(
            hidden_dim, num_heads, num_rbf, rbf_max, dropout
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        coordinates: torch.Tensor,
        valid: torch.Tensor,
        time_ids: torch.Tensor,
        atom_ids: torch.Tensor,
        pocket: torch.Tensor,
        pocket_ca: torch.Tensor,
        pocket_valid: torch.Tensor,
    ) -> torch.Tensor:
        tokens = tokens + self.joint(
            self.norm_joint(tokens), coordinates, valid, time_ids, atom_ids
        )
        tokens = tokens + self.cross(
            self.norm_cross(tokens),
            coordinates,
            valid,
            pocket,
            pocket_ca,
            pocket_valid,
        )
        return tokens + self.ffn(self.norm_ffn(tokens)) * valid.unsqueeze(-1)


class PocketEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.residue = nn.Embedding(32, hidden_dim, padding_idx=0)
        self.geometry = nn.Sequential(
            nn.Linear(5, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        layer = nn.TransformerEncoderLayer(
            hidden_dim,
            num_heads,
            hidden_dim * 4,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        n: torch.Tensor,
        ca: torch.Tensor,
        c: torch.Tensor,
        residue: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        n_vec, c_vec = n - ca, c - ca
        geometry = torch.stack(
            [
                n_vec.norm(dim=-1),
                c_vec.norm(dim=-1),
                (n - c).norm(dim=-1),
                (n_vec * c_vec).sum(dim=-1),
                torch.cross(n_vec, c_vec, dim=-1).norm(dim=-1),
            ],
            dim=-1,
        )
        token = self.residue((residue + 1).clamp_max(31)) + self.geometry(geometry)
        token = self.encoder(token, src_key_padding_mask=~mask)
        return self.norm(token) * mask.unsqueeze(-1)
