from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("data/train/ants_image/6240329_72c01e663e.jpg")

#ToTensor

trans_totensor = transforms.ToTensor()
image = trans_totensor(img)  #range[0,255]->[0.0,1.0]
writer.add_image("toTensor",image,1)
#Normalize
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_norm(image)
writer.add_image("Normalize",img_norm)

#Resize
print(img.size)
trans_resize = transforms.Resize((512,512))
#img_PIL ->resize ->img_resize PIL
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
print(img_resize)
#img_resize PIL ->totensor ->img_resize_tensor
img_resize_tensor = trans_resize(image)
print(img_resize_tensor)
writer.add_image("resize_tensor",img_resize_tensor)
writer.add_image("resize_PIL",img_resize)

#Compose -resize -2
trans_resize_2 = transforms.Resize(1024)
trans_compose = transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2 =  trans_compose(img)
writer.add_image("Resize",img_resize_2,2)

#RandomCrop
Trans_random = transforms.RandomCrop([128,256])
trans_compose_2 = transforms.Compose([Trans_random,trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop",img_crop,i)


writer.close()