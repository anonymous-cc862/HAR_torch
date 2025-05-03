

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hdim: int, nlayer: int = 1, activ: str= "Mish"):
        super().__init__()
        
        self.activ = getattr(nn, activ)()
        
        mid_l = [ self.activ]
        for _ in range(nlayer):
            mid_l += [nn.Linear(hdim, hdim),  self.activ]
            
        self.h_layer = nn.Sequential(*mid_l)
              
        self.in_layer =  nn.Linear(input_dim, hdim)
        self.out_layer = nn.Linear(hdim, output_dim)
    

    def forward(self, x: torch.Tensor):
        x = self.in_layer(x)
        x = self.h_layer(x)
        return self.out_layer(x)
    
    