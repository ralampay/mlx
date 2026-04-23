from __future__ import annotations

import torch
from torch import nn


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class LayerNorm2D(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class SelfAttention2D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        *,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm = nn.GroupNorm(1, dim)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.scale = self.head_dim ** -0.5
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        residual = x
        qkv = self.qkv(self.norm(x))
        q, k, v = torch.chunk(qkv, 3, dim=1)

        def reshape_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(batch_size, self.num_heads, self.head_dim, height * width)

        q = reshape_heads(q).transpose(-2, -1)
        k = reshape_heads(k)
        v = reshape_heads(v).transpose(-2, -1)

        attn = torch.matmul(q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(-2, -1).contiguous().reshape(batch_size, channels, height, width)
        out = self.proj(out)
        out = self.proj_dropout(out)
        return residual + out


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        expansion: int = 4,
        kernel_size: int = 7,
        layer_scale_init_value: float = 1e-6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if expansion < 1:
            raise ValueError("expansion must be at least 1")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve spatial size")

        hidden_dim = dim * expansion
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.norm = LayerNorm2D(dim)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.activation = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        else:
            self.layer_scale = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.activation(x)
        x = self.pwconv2(x)
        if self.layer_scale is not None:
            x = x * self.layer_scale.view(1, -1, 1, 1)
        x = self.dropout(x)
        return residual + x


def _resolve_num_heads(dim: int, max_heads: int = 8) -> int:
    for num_heads in range(min(max_heads, dim), 0, -1):
        if dim % num_heads == 0:
            return num_heads
    return 1


def _resolve_efficient_dim(dim: int) -> int:
    reduced_dim = max(32, dim // 2)
    while reduced_dim > 1 and dim % reduced_dim != 0:
        reduced_dim -= 1
    return reduced_dim


class DraxBlock(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        *,
        use_attention: bool = True,
        efficient: bool = True,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.efficient = efficient
        self.convnext = ConvNeXtBlock(dim)
        self.drop_path = DropPath(drop_path)

        if not use_attention:
            self.attention = None
            self.attn_down = None
            self.attn_up = None
            return

        attention_dim = _resolve_efficient_dim(dim) if efficient else dim
        self.attention = SelfAttention2D(attention_dim, num_heads=_resolve_num_heads(attention_dim))
        if efficient:
            self.attn_down = nn.Conv2d(dim, attention_dim, kernel_size=1)
            self.attn_up = nn.Conv2d(attention_dim, dim, kernel_size=1)
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

        fused_delta = 0.5 * (conv_delta + attention_delta)
        return x + self.drop_path(fused_delta)


class ConvActivationBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        activation_factory=nn.ReLU,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            activation_factory(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ConvActivationPoolBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        activation_factory=nn.ReLU,
        pool_factory=nn.MaxPool2d,
        pool_kernel_size: int = 2,
        pool_stride: int | None = None,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            ConvActivationBlock(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                activation_factory=activation_factory,
            ),
            pool_factory(pool_kernel_size, stride=pool_stride or pool_kernel_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
