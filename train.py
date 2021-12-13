import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from model import *
train_data = torchvision.datasets.CIFAR10("./data_set",train = True,transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.CIFAR10("./data_set",train = False,transform=torchvision.transforms.ToTensor(),download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
#如果train_data_size=10,训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

#利用DataLoader加载

train_dataloader = DataLoader(train_data,64)
test_dataloader = DataLoader(test_data,64)

#创建网络模型
Gration = gration()

#损失函数
loss__fn = nn.CrossEntropyLoss()

#优化器
#1e-2 = 1*(10）^2 = 0.01
learning_rate = 1e-2
optimizer = torch.optim.Adagrad(Gration.parameters(),lr = learning_rate)

#设置训练网络的一些参数
#记录训练的次数
total_train_step = 0
#记录测试的次数
total_test_step = 0
#训练的次数
epoch = 170
#添加tensorboard
writer = SummaryWriter("logs_train")

for i in range(epoch):
    print("-------------第{}轮训练开始-----------".format(i+1))


    #训练步骤开始
    best_loss = 1000000000
    Gration.train()#########非必要
    for data in train_dataloader:
        imgs,targets = data
        outputs = Gration(imgs)
        loss = loss__fn(outputs,targets)

        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step%100 == 0:
            print("训练次数：{}，loss：{}".format(total_train_step,loss.item()))  #item可以把数据拿出来
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    #测试步骤开始
    Gration.eval()########非必要 torch.nn查询
    total_test_loss = 0
    total_accuracy = 0

    with torch.no_grad(): ###比较有必要
        for data in test_dataloader:
            imgs,targets = data
            outputs = Gration(imgs)
            loss = loss__fn(outputs,targets)
            total_test_loss = total_test_loss + loss
            accuracy = (outputs.argmax(1)==targets).sum()
            total_accuracy = total_accuracy +accuracy

    if(total_test_loss>best_loss):
        best_loss = total_test_loss

    print("整体测试集上的loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy/test_data_size,))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)
    total_test_step = total_test_step + 1

    if(best_loss == total_test_loss):
        torch.save(Gration,"Gration.pth")
        print("模型已保存")


writer.close()