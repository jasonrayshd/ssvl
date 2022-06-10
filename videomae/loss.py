import torch
from torch import nn



class Ego4dTwoHead_Criterion(nn.Module):

    def __init__(self, criterion:nn.Module):
        super().__init__()
        self.criterion = criterion
    
    def forward(self, x: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        cls, loc = x
        label, state = target

        loc_loss = self.criterion(loc, label)
        cls_loss = self.criterion(cls, state)

        return cls_loss + loc_loss