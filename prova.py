import torch
from cosface_loss import MarginCosineProduct as cosface
from arcface_loss import MarginCosineProduct as arcface
from sphereface_loss import MarginCosineProduct as sphereface
import torch.nn as nn
import torch.nn.functional as F

class ArcFace(nn.Module):
    """ reference: <Additive Angular Margin Loss for Deep Face Recognition>
    """
    def __init__(self, feat_dim, num_class, s=64., m=0.5):
        super(ArcFace, self).__init__()
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.s = s
        self.m = m
        self.w = nn.Parameter(torch.Tensor(feat_dim, num_class))
        nn.init.xavier_normal_(self.w)
        #nn.init.constant_(self.w,3)

    def forward(self, x, y):
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)
        cos_theta = F.normalize(x, dim=1).mm(self.w)
        with torch.no_grad():
            theta_m = torch.acos(cos_theta.clamp(-1+1e-5, 1-1e-5))
            theta_m.scatter_(1, y.view(-1, 1), self.m, reduce='add')
            theta_m.clamp_(1e-5, 3.14159)
            d_theta = torch.cos(theta_m) - cos_theta

        logits = self.s * (cos_theta + d_theta)
        loss = F.cross_entropy(logits, y)
        return loss

for i in range(5):
    criterion = torch.nn.CrossEntropyLoss()

    my_cf = arcface(100, 3).double()
    
    test_tensor = torch.rand(1,100).double()
    logit = my_cf.forward(test_tensor, torch.tensor([[1]]))
    criterion(logit, torch.tensor([1]) ).backward()

    arc = ArcFace(100,3, s = 30, m = 0.40).double()
    (arc(test_tensor,torch.tensor([1]))).backward()




