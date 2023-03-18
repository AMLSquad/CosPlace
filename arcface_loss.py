import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F



def cosine_sim(x1: torch.Tensor, x2: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1, w2).clamp(min=eps)


class MarginCosineProduct(nn.Module):
   
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.40):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_normal_(self.weight)
    
    def forward(self, inputs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        cosine = cosine_sim(inputs, self.weight).clamp(-1+1e-5, 1-1e-5)
        cosine_plus_m = torch.clamp(torch.cos(torch.acos(cosine) + self.m),1e-5, 3.14159)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        my_cosine_vector = (one_hot * cosine_plus_m + (1.0-one_hot) * cosine)
        output = self.s * (my_cosine_vector)
        return output
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'



