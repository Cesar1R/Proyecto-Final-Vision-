import torch
import torch.nn as nn
from pytorchvideo.models.hub import i3d_r50

class R50(nn.Module):
    def __init__(self):
        super(R50, self).__init__()
        # Carga el I3D R50 oficial de pytorchvideo
        self.model = i3d_r50(pretrained=False) # Pretrained=False porque cargarás tus propios pesos
        self.blocks = self.model.blocks

    def forward(self, x):
        # I3D espera [B, C, T, H, W]
        return self.model(x)