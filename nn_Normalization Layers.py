import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[100,-500],
                      [-100,300]],dtype = torch.float32)

dataset = torchvision.datasets.CIFAR10("./data_set",False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,64,True)

class Gration(nn.Module):
    def __init__(self):
        super(Gration, self).__init__()
        self.batchNorm = nn.BatchNorm2d(1)
        self.relu1 = nn.ReLU()

    def forward(self,input):
        output = self.batchNorm(input)
        return output

input = torch.reshape(input,(1,1,2,2))
gration = Gration()
output = gration(input)
print(output)
# step = 0
# writer = SummaryWriter("./logs_ReLU")
# gration =Gration()
# for data in dataloader:
#     imgs,targets = data
#     writer.add_images("input", imgs, step)
#     output = gration(imgs)
#     writer.add_images("relu",output,step)
#     step +=1
#
# writer.close()