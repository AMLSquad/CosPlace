
# Based on https://github.com/MuggleWang/CosFace_pytorch/blob/master/layer.py

from typing import Optional
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.modules.loss import _WeightedLoss
from torch import Tensor
import torch.nn.functional as F

def cosine_sim(x1: torch.Tensor, x2: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1, w2).clamp(min=eps)


class MarginCosineProduct(nn.Module):
    """Implement of large margin cosine distance:
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """
    def __init__(self, in_features: int, out_features: int, l = 1, s: float = 30.0, m: float = 0.40):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.l = l
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    
    def forward(self, inputs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        #Cosine similarity between x and weights.
        cosine = cosine_sim(inputs, self.weight)
        one_hot = torch.zeros_like(cosine)
        #label.view => vettore colonna. 
        #Mette m solo dove c'è yi. Il resto rimane senza m. 
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = self.s * (cosine - one_hot * self.m)
        #output sul quale verrà applicata la cross entropy loss.
        SM = self.l * torch.mm(inputs, self.weight.t())
        output = output + SM
        return output
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


class NewLoss(_WeightedLoss):
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, ASM: Tensor, SM: Tensor, target: Tensor) -> Tensor:
        first_term =  F.cross_entropy(ASM, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction,
                               label_smoothing=self.label_smoothing)
        second_term = F.cross_entropy(SM, target, weight=self.weight,
                                      ignore_index=self.ignore_index, reduction=self.reduction,
                                    label_smoothing=self.label_smoothing)
        return first_term + second_term