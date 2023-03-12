import torch
from torch import nn
from torch.nn import functional as F

from timm.utils import accuracy
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


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


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self,  gamma=2):
        super(FocalLoss, self).__init__()

        self.celoss = nn.CrossEntropyLoss(reduction="none")
        self.gamma = gamma

    def forward(self, outputs, targets):
       
        """
            Args:
                outputs: B, N
                targets: B, N
        """

        B, *_ = targets.shape
        p = F.softmax(outputs, dim=1)
        p_class = [ p[i, targets[i]] for i in range(B) ]
        p_class = torch.stack(p_class, dim=0)

        ce_value = self.celoss(outputs, targets)
        loss = torch.pow((1-p_class), self.gamma)*ce_value

        return loss.mean()


class ActionAnticipationLoss(nn.Module):

    def __init__(self, celoss="focal", gamma=2, head_type="varant"):
        super().__init__()
        self.head_type=head_type
        if celoss == "focal":
            self.celoss = FocalLoss(gamma=gamma)
        elif celoss == "soft":
            self.celoss = SoftTargetCrossEntropy()
        else:
            self.celoss = nn.CrossEntropyLoss()

        self.mseloss = nn.MSELoss()
        self.mmloss = nn.MarginRankingLoss()
        self.l1_crit = nn.L1Loss()

    
    def forward(self, output, target):
        loss = 0.
        if self.head_type == "varant":
            out_cur, out_future, kld_obs_goal, kld_next_goal, kld_goal_diff, kld_future_goal, kld_future_goal_dis = output
            loss += kld_obs_goal
            loss += kld_next_goal  
            loss += kld_goal_diff
            loss += kld_future_goal
            loss += kld_future_goal_dis
        elif self.head_type == "baseline":
            out_cur, out_future = output[0], output[1:]

        next_act_loss = 0.
        for i, action in enumerate(target[1:]):
            next_act_loss += self.celoss(out_future[i], action)
        loss += next_act_loss
        cur_act_loss = self.celoss(out_cur, target[0])
        loss += cur_act_loss
 
        return loss

class HandsPredictionLoss(nn.Module):
    # smoother l1 loss
    def __init__(self, loss_type="l1", beta=5):
    
        super().__init__()
        if loss_type == "l1":
            self.l1_loss = torch.nn.SmoothL1Loss(reduction="sum",beta=5.0)
        else:
            raise NotImpelementedError(f"loss type: {loss_type} is not supported!")

    def forward(self, output, target_lst):
        target, mask = target_lst

        effective_num = mask.sum()
        loss = self.l1_loss(mask*output, target)
        loss /= effective_num

        return loss
    

class PNRLoss(nn.Module):

    def __init__(self, smoothing=0.1):

        super().__init__()

        self.bound = 2 # only using smoothing within certrain range around key frame

        self.smoothing = smoothing
        self.confidence = 1 - self.smoothing

    def forward(self, output, target):

        """
            output: (B, T+1)
            target: (B)
        """
        B, d = output.shape
        T = d - 1

        logprobs = F.log_softmax(output, dim=-1) # (B, T+1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss =  nll_loss.squeeze(1) # (B)
 
        sign = ( target < T ).long().cuda() # (B, 1) indicate whether state change occurs
        loc = torch.arange(-self.bound, self.bound+1, 1).unsqueeze(0).repeat(B, 1).cuda() + target.unsqueeze(1)
        mask = torch.where((loc>=0)*(loc<T), 1, 0)

        loc = torch.clamp(loc, min=0, max=T-1) # (B, self.bound*2+1)

        smooth_loss = -logprobs.gather(dim=-1, index=loc) # (B, self.bound*2+1)
        smooth_loss = smooth_loss * mask

        loss =  self.confidence * nll_loss  + sign*(self.smoothing * smooth_loss.mean(-1)) + (1-sign)*(self.smoothing * nll_loss)

        return loss.mean()


if __name__ == "__main__":

    loss_fn = PNRLoss()
    B = 4
    T = 16

    target = (torch.randint(0, T+1, size=(B, )))
    output = torch.randn(B,17)
    print(target)
    loss_fn(output, target)