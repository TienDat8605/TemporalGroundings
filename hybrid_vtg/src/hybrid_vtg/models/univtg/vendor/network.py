"""Inference-only UniVTG network with official checkpoint-compatible names."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn.functional as functional
from torch import nn

from .position_encoding import build_position_encoding
from .transformer import build_transformer


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def _mask_logits(inputs, mask, value: float = -1e30):
    return inputs + (1.0 - mask.float()) * value


def _absolute_sine_positions(values, dimension: int, temperature: float = 10000.0):
    """UniVTG's sine operator evaluated at normalized absolute video times."""
    scale = 2 * torch.pi
    frequencies = torch.arange(dimension, dtype=torch.float32, device=values.device)
    frequencies = temperature ** (2 * torch.div(frequencies, 2, rounding_mode="floor") / dimension)
    angles = values.float().clamp(0, 1).unsqueeze(-1) * scale / frequencies
    return torch.stack((angles[..., 0::2].sin(), angles[..., 1::2].cos()), dim=-1).flatten(-2)


class WeightedPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        weight = torch.empty(dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

    def forward(self, values, mask):
        alpha = torch.tensordot(values, self.weight, dims=1)
        alpha = nn.Softmax(dim=1)(_mask_logits(alpha, mask.unsqueeze(2)))
        return torch.matmul(values.transpose(1, 2), alpha).squeeze(2)


class Conv(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, layers: int) -> None:
        super().__init__()
        widths = [hidden_dim] * (layers - 1)
        self.layers = nn.ModuleList(
            nn.Conv1d(source, target, kernel_size=3, padding=1)
            for source, target in zip([input_dim, *widths], [*widths, output_dim])
        )

    def forward(self, values):
        values = values.permute(0, 2, 1)
        for index, layer in enumerate(self.layers):
            values = functional.relu(layer(values)) if index < len(self.layers) - 1 else layer(values)
        return values.permute(0, 2, 1)


class LinearLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, *, dropout: float, relu: bool) -> None:
        super().__init__()
        self.relu = relu
        self.layer_norm = True
        self.LayerNorm = nn.LayerNorm(input_dim)
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(input_dim, output_dim))

    def forward(self, values):
        values = self.net(self.LayerNorm(values))
        return functional.relu(values, inplace=True) if self.relu else values


@dataclass(frozen=True)
class NetworkSpec:
    video_dim: int
    text_dim: int = 512
    hidden_dim: int = 1024
    encoder_layers: int = 4
    input_projections: int = 2
    text_positions: bool = False
    maximum_video_length: int = 75


class UniVTGNetwork(nn.Module):
    def __init__(self, spec: NetworkSpec) -> None:
        super().__init__()
        args = SimpleNamespace(
            hidden_dim=spec.hidden_dim,
            dropout=0.0,
            droppath=0.0,
            nheads=8,
            dim_feedforward=2048,
            enc_layers=spec.encoder_layers,
            dec_layers=0,
            pre_norm=False,
            position_embedding="sine",
            max_q_l=32,
            input_dropout=0.0,
        )
        self.transformer = build_transformer(args)
        self.position_embed, self.txt_position_embed = build_position_encoding(args)
        self.span_loss_type = "l1"
        self.max_v_l = spec.maximum_video_length
        self.token_type_embeddings = nn.Embedding(2, spec.hidden_dim)
        self.token_type_embeddings.apply(_init_weights)
        self.span_embed = Conv(spec.hidden_dim, spec.hidden_dim, 2, 3)
        self.class_embed = Conv(spec.hidden_dim, spec.hidden_dim, 1, 3)
        self.use_txt_pos = spec.text_positions
        self.n_input_proj = spec.input_projections
        relu = [True] * 3
        relu[spec.input_projections - 1] = False
        self.input_txt_proj = nn.Sequential(*[
            LinearLayer(
                spec.text_dim if index == 0 else spec.hidden_dim,
                spec.hidden_dim,
                dropout=0.0,
                relu=relu[index],
            )
            for index in range(spec.input_projections)
        ])
        self.input_vid_proj = nn.Sequential(*[
            LinearLayer(
                spec.video_dim if index == 0 else spec.hidden_dim,
                spec.hidden_dim,
                dropout=0.0,
                relu=relu[index],
            )
            for index in range(spec.input_projections)
        ])
        self.weightedpool = WeightedPool(spec.hidden_dim)

    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, video_positions=None):
        src_vid = self.input_vid_proj(src_vid)
        src_txt = self.input_txt_proj(src_txt)
        src_vid = src_vid + self.token_type_embeddings(torch.ones_like(src_vid_mask.long()))
        src_txt = src_txt + self.token_type_embeddings(torch.zeros_like(src_txt_mask.long()))
        source = torch.cat([src_vid, src_txt], dim=1)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()
        video_position = (
            self.position_embed(src_vid, src_vid_mask)
            if video_positions is None
            else _absolute_sine_positions(video_positions, src_vid.shape[-1])
        )
        text_position = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)
        memory = self.transformer(source, ~mask, torch.cat([video_position, text_position], dim=1))
        video_memory = memory[:, :src_vid.shape[1], :]
        foreground = self.class_embed(video_memory).sigmoid()
        offsets = self.span_embed(video_memory).sigmoid()
        offsets = offsets * torch.tensor((-1, 1), device=offsets.device).view(1, 1, 2)
        sentence = self.weightedpool(src_txt, src_txt_mask).unsqueeze(1)
        saliency = functional.cosine_similarity(src_vid, sentence, dim=-1)
        return {"pred_logits": foreground, "pred_spans": offsets, "saliency_scores": saliency}
