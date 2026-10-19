"""Task-intent decoder over pooled DINOv3 frame features (shared by 29_mem_intent.py and the MEM-2 client)."""
import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, n_frames, n_cls, width=256, depth=4):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, width))
        self.pos = nn.Parameter(torch.randn(64, width) * 0.02)
        self.slot = nn.Parameter(torch.randn(n_frames, 1, width) * 0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(width, 4, width * 4, dropout=0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, depth)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, n_cls))

    def forward(self, x):  # [B, F, 64, 768]
        b = x.shape[0]
        h = self.inp(x) + self.pos + self.slot
        h = torch.cat([self.cls.expand(b, -1, -1), h.reshape(b, -1, h.shape[-1])], 1)
        return self.head(self.tf(h)[:, 0])
