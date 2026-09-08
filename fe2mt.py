"""FE2MT model for collaborative HSI--LiDAR classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FE2MT"]


class DB2DWT2D(nn.Module):
    """Two-dimensional Daubechies-2 discrete wavelet transform."""

    def __init__(self) -> None:
        super().__init__()

        low = torch.tensor(
            [-0.1294095226, 0.2241438680, 0.8365163037, 0.4829629131],
            dtype=torch.float32,
        )
        high = torch.tensor(
            [-0.4829629131, 0.8365163037, -0.2241438680, -0.1294095226],
            dtype=torch.float32,
        )

        ll = torch.outer(low, low)
        lh = torch.outer(high, low)
        hl = torch.outer(low, high)
        hh = torch.outer(high, high)
        kernel = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)

        self.register_buffer("kernel", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, channels, height, width = x.shape

        pad_bottom = 1 + height % 2
        pad_right = 1 + width % 2
        x = F.pad(x, (1, pad_right, 1, pad_bottom), mode="reflect")

        kernel = self.kernel.repeat(channels, 1, 1, 1)
        return F.conv2d(x, kernel, stride=2, groups=channels)


class PixelEmbedding(nn.Module):
    """Project per-pixel features into token embeddings."""

    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class SpectralSpatial3DEmbedding(nn.Module):
    """Extract local spectral-spatial HSI features with 3-D convolutions."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(
                1,
                8,
                kernel_size=(7, 3, 3),
                stride=(2, 1, 1),
                padding=(3, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(8),
            nn.GELU(),
            nn.Conv3d(
                8,
                16,
                kernel_size=(5, 3, 3),
                stride=(2, 1, 1),
                padding=(2, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(16),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(16, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv3d(x.unsqueeze(1))
        x = x.mean(dim=2)
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class WaveletEmbedding(nn.Module):
    """Extract and fuse low- and high-frequency wavelet features."""

    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.dwt = DB2DWT2D()

        self.low_proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(embed_dim),
        )
        self.high_proj = nn.Sequential(
            nn.Conv2d(in_channels * 3, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(embed_dim),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, output_size) -> torch.Tensor:
        x = self.dwt(x)
        batch_size, _, height, width = x.shape
        x = x.reshape(batch_size, self.in_channels, 4, height, width)

        low = x[:, :, 0]
        high = torch.cat([x[:, :, 1], x[:, :, 2], x[:, :, 3]], dim=1)

        low_features = self.low_proj(low)
        high_features = self.high_proj(high)
        x = self.fusion(torch.cat([low_features, high_features], dim=1))

        x = F.interpolate(
            x,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        return x.flatten(2).transpose(1, 2)


class HSIEmbedding(nn.Module):
    """Construct HSI tokens from spectral-spatial and frequency features."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        num_tokens = patch_size**2

        self.spectral_spatial_embedding = SpectralSpatial3DEmbedding(embed_dim)
        self.wavelet_embedding = WaveletEmbedding(in_channels, embed_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_size = x.shape[-2:]
        spectral_spatial_tokens = self.spectral_spatial_embedding(x)
        wavelet_tokens = self.wavelet_embedding(x, output_size)

        gate = self.fusion_gate(
            torch.cat([spectral_spatial_tokens, wavelet_tokens], dim=-1)
        )
        x = gate * spectral_spatial_tokens + (1.0 - gate) * wavelet_tokens
        x = self.fusion(x)

        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        return self.dropout(x + self.pos_embed)


class LiDAREmbedding(nn.Module):
    """Construct LiDAR tokens from spatial and frequency features."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        num_tokens = patch_size**2

        self.spatial_embedding = PixelEmbedding(in_channels, embed_dim)
        self.wavelet_embedding = WaveletEmbedding(in_channels, embed_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_size = x.shape[-2:]
        spatial_tokens = self.spatial_embedding(x)
        wavelet_tokens = self.wavelet_embedding(x, output_size)

        gate = self.fusion_gate(torch.cat([spatial_tokens, wavelet_tokens], dim=-1))
        x = gate * spatial_tokens + (1.0 - gate) * wavelet_tokens
        x = self.fusion(x)

        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        return self.dropout(x + self.pos_embed)


class SpatialSelfAttention(nn.Module):
    """Spatial self-attention with a feed-forward network."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + out)
        x = self.norm2(x + self.ffn(x))
        return x


class SpectralSelfAttention(nn.Module):
    """HSI spectral self-attention over transposed spatial tokens."""

    def __init__(
        self,
        num_tokens: int,
        embed_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(num_tokens, embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_proj = nn.Linear(embed_dim, num_tokens)
        self.norm = nn.LayerNorm(num_tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_token = x[:, :1]
        x = x[:, 1:].transpose(1, 2)
        residual = x

        x = self.input_proj(x)
        out, _ = self.attn(x, x, x, need_weights=False)
        out = self.output_proj(out)
        x = self.norm(residual + out).transpose(1, 2)

        return torch.cat([cls_token, x], dim=1)


class CrossModalAlignmentBlock(nn.Module):
    """Adaptively align spatial tokens between HSI and LiDAR."""

    def __init__(self, embed_dim: int, reduction: int = 4) -> None:
        super().__init__()
        hidden_dim = max(embed_dim // reduction, 16)

        self.hsi_norm = nn.LayerNorm(embed_dim)
        self.lidar_norm = nn.LayerNorm(embed_dim)
        self.mask_net = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.align_scale = nn.Parameter(torch.tensor(-2.2))

    def forward(
        self,
        hsi: torch.Tensor,
        lidar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hsi_residual = hsi
        lidar_residual = lidar

        hsi_norm = self.hsi_norm(hsi)
        lidar_norm = self.lidar_norm(lidar)
        difference = torch.abs(hsi_norm - lidar_norm)

        mask = self.mask_net(
            torch.cat([hsi_norm, lidar_norm, difference], dim=-1)
        )
        scale = torch.sigmoid(self.align_scale)
        delta = lidar_residual - hsi_residual

        hsi = hsi_residual + scale * mask * delta
        lidar = lidar_residual - scale * mask * delta
        return hsi, lidar


class BidirectionalCrossAttention(nn.Module):
    """Bidirectional cross-attention between HSI and LiDAR tokens."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.hsi_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.lidar_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.hsi_norm = nn.LayerNorm(embed_dim)
        self.lidar_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        hsi: torch.Tensor,
        lidar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hsi_input = hsi
        lidar_input = lidar

        hsi_out, _ = self.hsi_attn(
            query=hsi_input,
            key=lidar_input,
            value=lidar_input,
            need_weights=False,
        )
        lidar_out, _ = self.lidar_attn(
            query=lidar_input,
            key=hsi_input,
            value=hsi_input,
            need_weights=False,
        )

        hsi = self.hsi_norm(hsi_input + hsi_out)
        lidar = self.lidar_norm(lidar_input + lidar_out)
        return hsi, lidar


class ModalFusionBlock(nn.Module):
    """One multimodal Transformer encoding block of FE2MT."""

    def __init__(
        self,
        embed_dim: int,
        num_tokens: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hsi_spatial_attn = SpatialSelfAttention(
            embed_dim,
            num_heads,
            dropout,
        )
        self.lidar_spatial_attn = SpatialSelfAttention(
            embed_dim,
            num_heads,
            dropout,
        )
        self.cross_modal_align = CrossModalAlignmentBlock(embed_dim)
        self.spectral_attn = SpectralSelfAttention(
            num_tokens,
            embed_dim,
            num_heads,
            dropout,
        )
        self.cross_attn = BidirectionalCrossAttention(
            embed_dim,
            num_heads,
            dropout,
        )

    def forward(
        self,
        hsi: torch.Tensor,
        lidar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hsi = self.hsi_spatial_attn(hsi)
        lidar = self.lidar_spatial_attn(lidar)

        hsi_cls = hsi[:, :1]
        lidar_cls = lidar[:, :1]
        hsi_spatial = hsi[:, 1:]
        lidar_spatial = lidar[:, 1:]

        hsi_spatial, lidar_spatial = self.cross_modal_align(
            hsi_spatial,
            lidar_spatial,
        )

        hsi = torch.cat([hsi_cls, hsi_spatial], dim=1)
        lidar = torch.cat([lidar_cls, lidar_spatial], dim=1)

        hsi = self.spectral_attn(hsi)
        hsi, lidar = self.cross_attn(hsi, lidar)
        return hsi, lidar


class FE2MT(nn.Module):
    """Frequency-Enhanced Embedding Multimodal Transformer."""

    def __init__(
        self,
        hsi_channels: int = 144,
        lidar_channels: int = 1,
        num_classes: int = 15,
        patch_size: int = 16,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.patch_size = patch_size
        num_tokens = patch_size**2

        self.hsi_embedding = HSIEmbedding(
            hsi_channels,
            embed_dim,
            patch_size,
            dropout,
        )
        self.lidar_embedding = LiDAREmbedding(
            lidar_channels,
            embed_dim,
            patch_size,
            dropout,
        )
        self.blocks = nn.ModuleList(
            [
                ModalFusionBlock(
                    embed_dim,
                    num_tokens,
                    num_heads,
                    dropout,
                )
                for _ in range(depth)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, hsi: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        hsi = self.hsi_embedding(hsi)
        lidar = self.lidar_embedding(lidar)

        for block in self.blocks:
            hsi, lidar = block(hsi, lidar)

        hsi_cls = hsi[:, 0]
        lidar_cls = lidar[:, 0]

        center = self.patch_size // 2
        center_idx = 1 + center * self.patch_size + center
        hsi_center = hsi[:, center_idx]
        lidar_center = lidar[:, center_idx]

        features = torch.cat(
            [hsi_cls, lidar_cls, hsi_center, lidar_center],
            dim=-1,
        )
        return self.head(features)
