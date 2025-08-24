# ==============================================================================
#  SEA-ViT: Sea Surface Current Forecasting with Vision Transformers and BiGRU
#  ----------------------------------------------------------------------------
#  Author   : Teerapong Panboonyuen (Kao)
#  Paper    : "SEA-ViT: Forecasting Sea Surface Currents Using a Vision Transformer
#              and GRU-Based Spatio-Temporal Covariance Model"
#  Version  : 1.0
#  License  : MIT
# ==============================================================================

import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16

class SEAViT(nn.Module):
    def __init__(self, input_channels=2, hidden_dim=256, num_gru_layers=2, forecast_steps=1):
        super(SEAViT, self).__init__()
        self.vit = vit_b_16(pretrained=True)
        self.vit.conv_proj = nn.Conv2d(input_channels, 768, kernel_size=16, stride=16)

        self.bi_gru = nn.GRU(
            input_size=768,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, forecast_steps * input_channels)  # Forecast U and V
        )

        self.forecast_steps = forecast_steps
        self.input_channels = input_channels

    def forward(self, x):
        # x: [B, T, C, H, W] where C = 2 (U and V)
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        vit_feats = self.vit(x)  # [B*T, 768]
        vit_feats = vit_feats.view(B, T, -1)  # [B, T, 768]

        gru_out, _ = self.bi_gru(vit_feats)  # [B, T, 2*hidden_dim]
        last_out = gru_out[:, -1, :]  # take last timestep

        out = self.fc(last_out)  # [B, forecast_steps * C]
        return out.view(B, self.forecast_steps, self.input_channels)