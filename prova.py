import torch

import torch.nn as nn
import torch.nn.functional as F


def my_softmax(t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    softmax = F.softmax(t, dim=1)
    return torch.gather(softmax, 1, label.view(-1,1))

new_tensor = torch.load("sm.pth")

label = torch.load("label.pth")

x = new_tensor[0]
print(torch.exp(x[0]))
print(torch.exp(x[1]))
print((torch.sum(torch.exp(x))))

print( torch.exp(x[2]) / (torch.sum(torch.exp(x))))

#print(my_softmax(new_tensor, label).sum())
#torch.save(new_tensor, "soups_output/face_soup/prova.pth")




