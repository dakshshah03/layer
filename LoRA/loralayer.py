import torch
from torch import nn

class LoRALinear(nn.Module):
    def __init__(self, 
                 pretrained_layer: nn.Linear, 
                 r: int,
                 alpha: int = 32):
        super().__init__()
        self.pretrained_layer = pretrained_layer
        self.pretrained_layer.weight.requires_grad_(False)
        self.pretrained_layer.bias.requires_grad_(False)

        self.d: int = self.pretrained_layer.in_features
        self.k: int = self.pretrained_layer.out_features
        self.r: int = r 
        self.alpha = alpha
        self.scale: float = self.alpha / self.r
        
        self.lora_A: nn.Parameter = nn.Parameter(torch.empty(self.r, self.k))
        self.lora_B: nn.Parameter = nn.Parameter(torch.empty(self.d, self.r))
        
        nn.init.normal_(self.lora_A)
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        base_pred = self.pretrained_layer(x)
        delta_pred = self.scale * x @ (self.lora_B @ self.lora_A)
        return base_pred + delta_pred

class LoRAAttention(nn.Module):
    pass

class LoRAConvolution(nn.Module):
    pass