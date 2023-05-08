
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

def my_softmax(t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    softmax = F.softmax(t, dim=1)
    return torch.gather(softmax, 1, label.view(-1,1))
    

class EMMS(nn.Module):
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
        # sposto la normalizzazione qui
       
        normalized_inputs = F.normalize(inputs, p=2.0, dim=1)
        cosine = cosine_sim(normalized_inputs, self.weight)
        
        one_hot = torch.zeros_like(cosine)
        #label.view => vettore colonna. 
        #Mette m solo dove c'è yi. Il resto rimane senza m. 
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = self.s * (cosine - one_hot * self.m) # batch x dimensione output
        softmax_cosface = my_softmax(output, label)

        SM = torch.mm(inputs, self.weight.t())
        
        softmax_SM = my_softmax(SM, label)
        sma = softmax_cosface + self.l * softmax_SM
        output =torch.log(sma)
    
        return output
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


class Mean():

    def __init__(self):
        super().__init__()

    def __call__(self, output: Tensor) -> Tensor:
        
        
        return torch.mean(output)