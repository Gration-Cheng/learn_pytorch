from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import cv2
#python 用法》tensor数据类型
#通过transform.ToTensor去看两个问题
# 1、transforms该如何使用（python）
# 2、为什么需要Tensor数据类型(包含了一些神经网络参数的张量)


img_path = "data/train/ants_image/5650366_e22b7e1065.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
writer.add_image("Tensor_img",tensor_img)

writer.close()