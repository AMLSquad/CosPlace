import torch

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
a.grad = torch.tensor(1.0)


c = a * b
c.backward(retain_graph=True)
c.backward()



