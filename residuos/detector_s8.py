"""Inference architecture identical to the existing trained DetectorS8."""
import torch
from torch import nn
from . import CLASSES
GRID=32
SLOTS=3

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(4, channels)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(4, channels)

    def forward(self, x):
        return x + self.gn2(self.conv2(self.act(self.gn1(self.conv1(x)))))


class DetectorS8(nn.Module):
    """Compact CNN detector designed from scratch with stride 8 (32x32 grid).
    Input: (B, 3, 256, 256)
    Output: (B, 32, 32, SLOTS, 11) where 11 = 1 presence + 4 box (dx, dy, w, h) + 6 class logits.
    """
    def __init__(self, slots=SLOTS, num_classes=len(CLASSES)):
        super().__init__()
        self.slots = slots
        self.num_classes = num_classes
        self.entry = nn.Sequential(
            # Stage 1: stride 2 -> 128x128
            nn.Conv2d(3, 24, 3, stride=2, padding=1),
            nn.GroupNorm(4, 24),
            nn.SiLU(),
            # Stage 2: stride 2 -> 64x64
            nn.Conv2d(24, 48, 3, stride=2, padding=1),
            nn.GroupNorm(4, 48),
            nn.SiLU(),
            ResBlock(48),
            # Stage 3: stride 2 -> 32x32
            nn.Conv2d(48, 96, 3, stride=2, padding=1),
            nn.GroupNorm(4, 96),
            nn.SiLU(),
            ResBlock(96),
            # Spatial refinement at 32x32 resolution
            nn.Conv2d(96, 96, 3, padding=1),
            nn.GroupNorm(4, 96),
            nn.SiLU(),
        )
        out_ch = slots * (5 + num_classes)
        self.head = nn.Conv2d(96, out_ch, 1)
        # Strict local initialization from scratch
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias.view(slots, -1)[:, 0] = -3.5

    def forward(self, x):
        z = self.head(self.entry(x))
        # z: (B, slots*(5+C), 32, 32)
        return z.view(len(x), self.slots, 5 + self.num_classes, GRID, GRID).permute(0, 3, 4, 1, 2)


