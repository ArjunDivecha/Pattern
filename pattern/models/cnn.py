"""
=============================================================================
SCRIPT NAME: cnn.py
=============================================================================
DESCRIPTION:
Parametric CNN for PRD §6.  Supports I5/I20/I60 via config.
FC input size is computed dynamically by running a dummy forward pass,
avoiding hard-coded spatial flow numbers.

Usage:
    from pattern.config import ModelConfig, ImageConfig
    model = ChartCNN(ModelConfig(), ImageConfig())
=============================================================================
"""

import torch
import torch.nn as nn

from pattern.config import ImageConfig, ModelConfig
from pattern.models.blocks import ConvBlock


class ChartCNN(nn.Module):
    """
    Chart image CNN (PRD §6).

    Architecture:
      `blocks` × ConvBlock (channels: 1 → channels[0] → ... → channels[-1])
      Flatten → Dropout(fc_dropout) → Linear(fc_in, 2)

    Weights initialised with Xavier uniform; biases zero.
    Softmax is NOT applied here — use nn.CrossEntropyLoss (which expects logits).
    For inference, apply softmax to get P(down), P(up).
    """

    def __init__(self, model_cfg: ModelConfig, image_cfg: ImageConfig):
        super().__init__()

        in_ch    = 1
        channels = model_cfg.channels[: model_cfg.blocks]
        blocks   = []
        for block_idx, out_ch in enumerate(channels):
            # Paper §II.B: asymmetric vertical stride + dilation exist only to
            # compress the sparse first layer. Subsequent blocks use stride=1
            # and dilation=1 with same-padding.
            if block_idx == 0:
                stride   = tuple(model_cfg.conv_stride)
                padding  = tuple(model_cfg.conv_padding)
                dilation = tuple(model_cfg.conv_dilation)
            else:
                stride   = tuple(model_cfg.conv_stride_inner)
                padding  = tuple(model_cfg.conv_padding_inner)
                dilation = tuple(model_cfg.conv_dilation_inner)

            blocks.append(ConvBlock(
                in_channels   = in_ch,
                out_channels  = out_ch,
                conv_kernel   = tuple(model_cfg.conv_kernel),
                conv_stride   = stride,
                conv_padding  = padding,
                conv_dilation = dilation,
                pool_kernel   = tuple(model_cfg.pool_kernel),
                leaky_slope   = model_cfg.leaky_slope,
            ))
            in_ch = out_ch

        self.conv_blocks = nn.Sequential(*blocks)
        self.dropout     = nn.Dropout(p=model_cfg.fc_dropout)

        # Compute fc input size with a dummy forward pass
        with torch.no_grad():
            dummy  = torch.zeros(1, 1, image_cfg.height, image_cfg.width)
            fc_in  = self.conv_blocks(dummy).numel()

        self.fc = nn.Linear(fc_in, 2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) float tensor, pixel values normalised to ~N(0,1).
        Returns:
            (B, 2) logit tensor — [logit_down, logit_up].
        """
        x = self.conv_blocks(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        return self.fc(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, 2) softmax probabilities [P(down), P(up)]."""
        return torch.softmax(self.forward(x), dim=1)

    def forward_with_features(self, x: torch.Tensor):
        """Return (logits, embedding). Embedding = global-avg-pool over the last
        conv block's spatial dims, giving a (B, C_last) compact stock representation
        (256-d for I20) suitable for similarity/regime analysis."""
        feat = self.conv_blocks(x)                          # (B, C, H', W')
        emb  = feat.mean(dim=(2, 3))                        # (B, C)
        flat = feat.flatten(start_dim=1)
        logits = self.fc(self.dropout(flat))
        return logits, emb
