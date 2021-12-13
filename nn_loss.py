import torch
from torch import nn
inputs = torch.tensor([1,2,3],dtype = torch.float32)
targets = torch.tensor([1,2,5],dtype = torch.float32)

inputs = torch.reshape(inputs,(1,1,1,3))
targets = torch.reshape(targets,(1,1,1,3))

loss1 = nn.L1Loss()
loss2 = nn.MSELoss()
result1 = loss1(inputs,targets)
result2 = loss2(inputs,targets)
print(result1)
print(result2)

x = torch.tensor([0.1,0.2,0.3])
y = torch.tensor([1])
x = torch.reshape(x,(1,3))
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x,y)
print(result_cross)