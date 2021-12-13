import torch
from torch import nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./data_set",train = False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,64)

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


loss =nn.CrossEntropyLoss()
gration = Gration()
optim = torch.optim.Adagrad(gration.parameters(),lr=0.01)
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs,targets = data
        output = gration(imgs)
        result_loss = loss(output,targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss = running_loss + result_loss
    print(running_loss)