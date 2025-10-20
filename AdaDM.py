import torch
import torch.nn as nn
from monai.networks.blocks import ResidualUnit as MONAIResidualUnit



    


    
class ResidualUnitAdaDM(MONAIResidualUnit):
    def __init__(self, *args, spatial_dims=3, **kwargs):
        super().__init__(*args, **kwargs)
        Conv = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        self.phi = Conv(1, 1, kernel_size=1, bias=True)
        with torch.no_grad():
            self.phi.weight.fill_(1.0)
            self.phi.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = list(range(1, x.ndim))
        s = torch.std(x, dim=dims, keepdim=True).clamp_min(1e-6)
        scale = torch.exp(self.phi(torch.log(s)))
        res = self.residual(x)
        cx  = self.conv(x)
        return cx * scale + res