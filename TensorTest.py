import torch
from torch.autograd.functional import hessian
  
# defining a function
def func(x):
    return (2*x.pow(3) - x.pow(2)).sum()
  
# defining the input tensor
x = torch.tensor([[2., 3., 4.],[1 ,2, 3]])
print(x)