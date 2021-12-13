import torchvision
from torch import nn
# train_data = torchvision.datasets.ImageNet("./data_image_net",split="train",download=True,
#                                            transform=torchvision.transforms.ToTensor())

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_ture = torchvision.models.vgg16(pretrained=True)
print(vgg16_ture)

train_data = torchvision.datasets.CIFAR10("./data_set",train=True,transform=torchvision.transforms.ToTensor(),
                                          download=True)
vgg16_ture.classifier.add_module('add_linear',nn.Linear(1000,10))
print(vgg16_ture)

vgg16_false.classifier[6] = nn.Linear(4096,10)  ##修改模型
print(vgg16_false)