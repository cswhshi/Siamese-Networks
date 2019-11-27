# -*- coding: utf-8 -*-
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dset
from SiameseNetworkDataset import SiameseNetworkDataset
from Model import SiameseNetwork

from VGG import VGG

from ContrastiveLoss import ContrastiveLoss

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()
    
    
training_dir = "./data/faces/training/"#训练数据集的路径
testing_dir = "./data/faces/testing/"#测试数据集的路径
model_path='./weight/net.path'
loss_csv='./weight/loss.csv'

batch_sizes=64

#定义训练集的dataset
train_dataset = SiameseNetworkDataset(imageFolderDataset=dset.ImageFolder(training_dir),
                    transform=transforms.Compose([transforms.Resize((100,100)),
                                                  transforms.ToTensor()
                                                  ])
                   ,should_invert=False)
    
test_dataset = SiameseNetworkDataset(imageFolderDataset=dset.ImageFolder(testing_dir),
                    transform=transforms.Compose([transforms.Resize((100,100)),
                                                  transforms.ToTensor()
                                                  ])
                   ,should_invert=False)
    
    
#训练集的dataloader
train_dataloader = DataLoader(train_dataset,shuffle=True,num_workers=1,batch_size=batch_sizes)
test_dataloader = DataLoader(test_dataset,num_workers=1,batch_size=1,shuffle=True)


#X_example,y_example,label=next(iter(train_dataloader))
#X_example.shape

net = SiameseNetwork()   #定义网络
net = VGG()   #定义网络
criterion = ContrastiveLoss()#定义损失函数
optimizer = optim.Adam(net.parameters(),lr = 0.0005 )#定义优化器
if torch.cuda.is_available():
    net=net.cuda()
    criterion=criterion.cuda()
    
counter = []
loss_history = [] 
distance_history=[]
iteration_number= 0

for epoch in range(0,10):
    ## 训练
    for i, data in enumerate(train_dataloader,0):
        img0, img1 ,label = data
        if torch.cuda.is_available():
            img0 = img0.cuda()
            img1 = img1.cuda()
            label = label.cuda()
        #img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
        optimizer.zero_grad()
        output1,output2 = net(img0,img1)
        
        distance = F.pairwise_distance(output1, output2)
        distance_history.append(distance)
        loss_contrastive = criterion(output1,output2,label)
        loss_contrastive.backward()
        optimizer.step()
        if i %10 == 0 :
            print("Epoch number {}, Current loss: {} , Distance: {}\n".format(epoch,loss_contrastive.item(),distance.mean().item()))
            #print("Epoch number {}, Current loss: {} \n".format(epoch,loss_contrastive.item()))
            iteration_number +=10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())
    #break
    torch.save(net.state_dict(), model_path) 


show_plot(counter,loss_history)

import pandas as pd
data=pd.DataFrame([counter,loss_history]).T
data.to_csv(loss_csv)
show_plot(data[0],data[1])