import torch

#方式1-》保存方式1，加载模型
import torchvision
from torch import nn
from model_save import gration
model = torch.load("vgg16_method1.pth")
# print(model)
#方式2，加载模型
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model2 = torch.load("vgg16_method2.pth")
# print(vgg16)

#陷阱1
# class gration(nn.Module):
#     def __init__(self):
#         super(gration, self).__init__()
#         self.conv1 = nn.Conv2d(3,64,3)
#
#     def forward(self,x):
#         x = self.conv1(x)
#         return x
model = torch.load("Gration_method1.pth")
print(model)