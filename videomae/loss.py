import torch
from torch import nn



class Ego4dTwoHead_Criterion(nn.Module):

    def __init__(self, criterion:nn.Module):
        super().__init__()
        self.criterion = criterion
    
    def forward(self, x: torch.Tensor, target, **kwargs) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            loc, cls = x

        if isinstance(target, torch.Tensor):
            return self.criterion(x, target.long())
        else:
            label, state = target[0].long(), target[1].long()

            loc_loss = self.criterion(loc, label)
            cls_loss = self.criterion(cls, state)

            return cls_loss + loc_loss
