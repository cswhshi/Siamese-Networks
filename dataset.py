# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets,transforms,models
from PIL import Image,ImageOps
import random
import torch
import matplotlib.pyplot as plt
import numpy as np







class SiameseNetworkDataset(Dataset):
    def __init__(self,imageFolderDataset,transform=None,should_invert=True):
        self.imageFolderDataset=imageFolderDataset
        self.transform=transform
        self.should_invert=should_invert
    def __getitem__(self,index):
        """
        [('./data/faces/training/s1\\1.pgm', 0),
         ('./data/faces/training/s1\\10.pgm', 0)]
        """
        img0_tuple=random.choice(self.imageFolderDataset.imgs)#随机获取一张图片文件以及对应的标签
        should_get_same_class = random.randint(0,1)  #用户选取是否同一个类别的图片
        if(should_get_same_class):
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs) #随机选择第二个图片以及对应的标签
                if img0_tuple[1]==img1_tuple[1]:#如果随机选取的图片标签是相同的，则跳出循环
                    break
        else:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] !=img1_tuple[1]:
                    break
        
        img0 = Image.open(img0_tuple[0])#读取第一张图片
        img1 = Image.open(img1_tuple[0])#读取第二章图片
        
        img0 = img0.convert("L")#彩色图像转灰度图
        img1 = img1.convert("L")
        
        
        if self.should_invert:#是否进行色彩反转
            img0 = ImageOps.invert(img0)
            img1 = ImageOps.invert(img1)
        if self.transform is not None:      #是否进行数据预处理
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        #返回两种选择的图片，以及它们是否是同一个类别
        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
    def __len__(self):
        return len(self.imageFolderDataset.imgs) 


import os
root_dir = os.path.abspath('.') ## 获取相对目录

training_dir = os.path.join(root_dir ,"data","train")
folder_dataset = datasets.ImageFolder(training_dir)

train_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                    transform=transforms.Compose([transforms.Resize((100,100)),
                                                  transforms.ToTensor()
                                                  ])
                   ,should_invert=False)

X_example,y_example,label=next(iter(train_dataset))
print(X_example.shape,y_example.shape,label)

import torchvision
img=torchvision.utils.make_grid(X_example)
img=img.numpy().transpose([1,2,0])
plt.imshow(img)
plt.show()
    
img1=torchvision.utils.make_grid(y_example)
img1=img1.numpy().transpose([1,2,0])
plt.imshow(img1)
plt.show()
    
print(label)

