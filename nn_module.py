import torch
from torch import nn

class Gration(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input):
        output = input+1
        return output

gration = Gration()
x= torch.tensor(1.0)
output = gration(x)
print(output)
