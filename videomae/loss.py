import torch
from torch import nn
from timm.utils import accuracy


class OsccPNRCriterion(nn.Module):

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


# class OsccCriterion(nn.Module):

#     def __init__(self, criterion:nn.Module):
#         super().__init__()
#         self.criterion = criterion

#     def forward(self, x: torch.Tensor, target, **kwargs) -> torch.Tensor:

#         state = target[1]

#         return self.criterion(x, state.long())


# class PNRCriterion(nn.Module):

#     def __init__(self, criterion:nn.Module):
#         super().__init__()
#         self.criterion = criterion

#     def forward(self, x: torch.Tensor, target, **kwargs) -> torch.Tensor:

#         label = target[0]
#         return self.criterion(x, label.long())


class ActionAnticipationLoss(nn.Module):

    def __init__(self, task=""):
        super().__init__()
        self.celoss = nn.CrossEntropyLoss()
        self.mseloss = nn.MSELoss()
        self.mmloss = nn.MarginRankingLoss()
        self.l1_crit = nn.L1Loss()

        self.task = task
    
    def forward(self, output, target):

        verb, noun = target
        correct_action = 0
        device = output.device

        if self.task == "lta_verb":
            next_action = torch.LongTensor([ int(verb[1]) ]).to(device)
            cur_action = torch.LongTensor([ int(verb[0]) ]).to(device)
        elif self.task == "lta_noun":
            next_action = torch.LongTensor([ int(noun[1]) ]).to(device)
            cur_action = torch.LongTensor([ int(noun[0]) ]).to(device)

        out_next, out_cur, kld_obs_goal, kld_next_goal, kld_goal_diff = output
        # pred_next = torch.argmax(out_next,1)
        # pred_cur = torch.argmax(out_cur,1)

        # print('out_next:', out_next.shape)    
        # print('out_cur:', out_cur.shape)    
        # print('next_action:', next_action)
        # Next action loss

        if next_action.item() != -1:
            next_act_loss = self.celoss(out_next, next_action)
            loss = next_act_loss

        loss += kld_obs_goal
        loss += kld_next_goal  

        if cur_action.item() != -1:
            cur_act_loss = self.celoss(out_cur, cur_action)
            loss += cur_act_loss

        loss += kld_goal_diff
 
        return loss
    