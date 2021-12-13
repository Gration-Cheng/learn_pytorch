import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./data_set",train = False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset,batch_size=64,drop_last=True)

class Gration(nn.Module):
    def __init__(self):
        super(Gration, self).__init__()
        self.linear1 = nn.Linear(196608,1)

    def forward(self,input):
        output = self.linear1(input)
        return output

gration = Gration()
for data in dataloader:
    imgs,targets = data
    output = torch.flatten(imgs)
    output = gration(output)
    print(output)