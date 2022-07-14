import torch
from torch import nn



class Ego4dTwoHead_Criterion(nn.Module):

    def __init__(self, criterion:nn.Module, lamb_cls=1, lamb_loc=1):
        super().__init__()
        self.criterion = criterion
        self.lamb_cls = lamb_cls
        self.lamb_loc = lamb_loc

    def forward(self, x: torch.Tensor, target, **kwargs) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            loc, cls = x

        if isinstance(target, torch.Tensor):
            return self.criterion(x, target.long())
        else:

            label, state = target

            loc_loss = self.criterion(loc, label)
            cls_loss = self.criterion(cls, state)

            return self.lamb_cls*cls_loss + self.lamb_loc*loc_loss
