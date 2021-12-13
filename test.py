##test就是运行到实际环境中

import torchvision.transforms
from PIL import Image
from torch import nn
import torch
image_path = "./dog/002.png"
Image = Image.open(image_path)
print(Image)
device = torch.device("cuda")
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                           torchvision.transforms.ToTensor()])

Image = transform(Image)
class gration(nn.Module):
    def __init__(self):
        super(gration, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024,64),
            nn.Linear(64,10)
        )

    def forward(self,x):
        x = self.model(x)
        return x

model = torch.load("./Gration.pth")
model = model.to(device)
Image = torch.reshape(Image,(1,3,32,32))
Image = Image.to(device)
model.eval()
with torch.no_grad():
    output = model(Image)
print(output)
print(output.argmax(1))