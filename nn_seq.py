import torch
from torch import nn
import torchvision
from torch.utils.tensorboard import SummaryWriter


class Gration(nn.Module):
    def __init__(self):
        super(Gration, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3,32,5,padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,2,padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,padding = 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024,64),
            nn.Linear(64,10),
        )

    def forward(self,x):
        x = self.model1(x)
        return x

gration = Gration()
print(gration)
input = torch.ones((64,3,32,32))
output = gration(input)
print(output.shape)


writer = SummaryWriter("./logs_seq")
writer.add_graph(gration,input)
writer.close()