from __future__ import annotations

import copy

import torch
from torch import nn

from mlx.modes.image_classification.models.blocks import (
    DropPath,
    DraxBlock,
    resolve_attention_num_heads,
    resolve_drax_fusion_mode,
    resolve_efficient_attention_dim,
)


class SelfAttention3D(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm = nn.GroupNorm(1, dim)
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv3d(dim, dim, kernel_size=1)
        self.scale = self.head_dim**-0.5
        self.attn_dropout = nn.Dropout()
        self.proj_dropout = nn.Dropout()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = x.shape
        residual = x
        qkv = self.qkv(self.norm(x))
        q, k, v = torch.chunk(qkv, 3, dim=1)
        tokens = frames * height * width

        def reshape_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(batch, self.num_heads, self.head_dim, tokens)

        q = reshape_heads(q).transpose(-2, -1)
        k = reshape_heads(k)
        v = reshape_heads(v).transpose(-2, -1)
        attention = (torch.matmul(q, k) * self.scale).softmax(dim=-1)
        attention = self.attn_dropout(attention)
        output = torch.matmul(attention, v)
        output = output.transpose(-2, -1).contiguous().reshape(
            batch, channels, frames, height, width
        )
        return residual + self.proj_dropout(self.proj(output))


class LayerNorm3D(nn.Module):
    """Apply channel-wise layer normalization to a ``[B, C, T, H, W]`` tensor."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        return x.permute(0, 4, 1, 2, 3)


class ConvNeXtBlock3D(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        temporal_kernel_size: int,
        expansion: int = 4,
        spatial_kernel_size: int = 7,
        layer_scale_init_value: float = 1e-6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = dim * expansion
        self.dwconv = nn.Conv3d(
            dim,
            dim,
            kernel_size=(temporal_kernel_size, spatial_kernel_size, spatial_kernel_size),
            padding=(temporal_kernel_size // 2, spatial_kernel_size // 2, spatial_kernel_size // 2),
            groups=dim,
        )
        self.norm = LayerNorm3D(dim)
        self.pwconv1 = nn.Conv3d(dim, hidden_dim, kernel_size=1)
        self.activation = nn.GELU()
        self.pwconv2 = nn.Conv3d(hidden_dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.layer_scale = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim))
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv2(self.activation(self.pwconv1(x)))
        if self.layer_scale is not None:
            x = x * self.layer_scale.view(1, -1, 1, 1, 1)
        return residual + self.dropout(x)


class DraxBlock3D(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        temporal_kernel_size: int = 3,
        use_attention: bool = True,
        efficient: bool = True,
        drop_path: float = 0.0,
        fusion_mode: str = "average",
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.efficient = efficient
        self.fusion_mode = resolve_drax_fusion_mode(fusion_mode)
        self.convnext = ConvNeXtBlock3D(
            dim, temporal_kernel_size=temporal_kernel_size
        )
        self.drop_path = DropPath(drop_path)

        if use_attention and self.fusion_mode == "sknet":
            fusion_dim = max(32, dim // 16)
            self.fusion_gate = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(dim, fusion_dim, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(fusion_dim, 2 * dim, kernel_size=1),
            )
        else:
            self.fusion_gate = None

        if not use_attention:
            self.attention = None
            self.attn_down = None
            self.attn_up = None
            return
        attention_dim = resolve_efficient_attention_dim(dim) if efficient else dim
        self.attention = SelfAttention3D(
            attention_dim, num_heads=resolve_attention_num_heads(attention_dim)
        )
        if efficient:
            self.attn_down = nn.Conv3d(dim, attention_dim, kernel_size=1)
            self.attn_up = nn.Conv3d(attention_dim, dim, kernel_size=1)
        else:
            self.attn_down = None
            self.attn_up = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_delta = self.convnext(x) - x
        if not self.use_attention or self.attention is None:
            return x + self.drop_path(conv_delta)
        if self.efficient:
            reduced = self.attn_down(x)
            attention_delta = self.attn_up(self.attention(reduced) - reduced)
        else:
            attention_delta = self.attention(x) - x
        if self.fusion_mode == "average":
            fused = 0.5 * (conv_delta + attention_delta)
        else:
            if self.fusion_gate is None:
                raise RuntimeError("SKNet fusion requires an initialized fusion gate.")
            batch, channels, _, _, _ = conv_delta.shape
            logits = self.fusion_gate(conv_delta + attention_delta)
            weights = logits.reshape(batch, 2, channels, 1, 1, 1).softmax(dim=1)
            fused = weights[:, 0] * conv_delta + weights[:, 1] * attention_delta
        return x + self.drop_path(fused)


def inflate_drax_block(source: DraxBlock, temporal_kernel_size: int) -> DraxBlock3D:
    target = DraxBlock3D(
        source.convnext.dwconv.in_channels,
        temporal_kernel_size=temporal_kernel_size,
        use_attention=source.use_attention,
        efficient=source.efficient,
        drop_path=source.drop_path.drop_prob,
        fusion_mode=source.fusion_mode,
    )
    _copy_convnext(source.convnext, target.convnext, temporal_kernel_size)
    if source.fusion_gate is not None and target.fusion_gate is not None:
        _copy_conv(source.fusion_gate[1], target.fusion_gate[1], 1)
        _copy_conv(source.fusion_gate[3], target.fusion_gate[3], 1)
    if source.attention is not None and target.attention is not None:
        target.attention.norm.load_state_dict(source.attention.norm.state_dict())
        _copy_conv(source.attention.qkv, target.attention.qkv, 1)
        _copy_conv(source.attention.proj, target.attention.proj, 1)
        target.attention.attn_dropout = copy.deepcopy(source.attention.attn_dropout)
        target.attention.proj_dropout = copy.deepcopy(source.attention.proj_dropout)
    if source.attn_down is not None and target.attn_down is not None:
        _copy_conv(source.attn_down, target.attn_down, 1)
    if source.attn_up is not None and target.attn_up is not None:
        _copy_conv(source.attn_up, target.attn_up, 1)
    return target


def inflate_convnext_block(
    source, temporal_kernel_size: int
) -> ConvNeXtBlock3D:
    target = ConvNeXtBlock3D(
        source.dwconv.in_channels,
        temporal_kernel_size=temporal_kernel_size,
    )
    _copy_convnext(source, target, temporal_kernel_size)
    return target


def _copy_convnext(source, target, temporal_kernel_size: int) -> None:
    _copy_conv(source.dwconv, target.dwconv, temporal_kernel_size)
    target.norm.norm.load_state_dict(source.norm.norm.state_dict())
    _copy_conv(source.pwconv1, target.pwconv1, 1)
    _copy_conv(source.pwconv2, target.pwconv2, 1)
    target.dropout = copy.deepcopy(source.dropout)
    if source.layer_scale is not None and target.layer_scale is not None:
        target.layer_scale.data.copy_(source.layer_scale.data)


def _copy_conv(source: nn.Conv2d, target: nn.Conv3d, temporal_kernel_size: int) -> None:
    with torch.no_grad():
        target.weight.copy_(
            source.weight.unsqueeze(2).repeat(1, 1, temporal_kernel_size, 1, 1)
            / temporal_kernel_size
        )
        if source.bias is not None and target.bias is not None:
            target.bias.copy_(source.bias)


__all__ = [
    "ConvNeXtBlock3D",
    "DraxBlock3D",
    "LayerNorm3D",
    "SelfAttention3D",
    "inflate_convnext_block",
    "inflate_drax_block",
]
