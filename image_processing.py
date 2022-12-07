#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
get_ipython().magic(u'matplotlib inline')
import torchvision
import torchvision.datasets as dset
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os

class CustomImageDataset(Dataset):
    def read_data_set(self):
        all_img_files = []
        all_labels = []

        class_names = os.listdir(self.data_set_path)
        
        for index, class_name in enumerate(class_names):
            if class_name !='.ipynb_checkpo':
                label = class_name
                img_dir = os.path.join(self.data_set_path, class_name)
                img_files = os.listdir(img_dir)

                for img_file in img_files:
                    if img_file !='.ipynb_checkpoints':
                        img_file = os.path.join(img_dir, img_file)
                        img = Image.open(img_file)
                        if img is not None:
                            all_img_files.append(img_file)
                            all_labels.append(int(label))

        return all_img_files, all_labels, len(all_img_files), len(class_names)

    def __init__(self, data_set_path, transforms=None):
        self.data_set_path = data_set_path
        self.image_files_path, self.labels, self.length, self.num_classes = self.read_data_set()
        self.transforms = transforms

    def __getitem__(self, index):
        image = Image.open(self.image_files_path[index])
        image = image.convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)

        return {'image': image, 'label': self.labels[index]}

    def __len__(self):
        return self.length

train_data_path = './data/train_img/'   
test_data_path = './data/test_img/'  
    

hyper_param_epoch = 20
hyper_param_batch = 8
hyper_param_learning_rate = 0.001
transforms_train = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.RandomRotation(10.),
                                       transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transforms_test = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data_set = CustomImageDataset(data_set_path= train_data_path, transforms=transforms_train)
train_loader = DataLoader(train_data_set, batch_size=128, shuffle=True)

test_data_set = CustomImageDataset(data_set_path=test_data_path, transforms=transforms_test)
test_loader = DataLoader(test_data_set, batch_size=1, shuffle=False)

class CustomConvNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomConvNet, self).__init__()
        self.layer1 = self.conv_module(3, 16)
        self.layer2 = self.conv_module(16, 32)
        self.layer3 = self.conv_module(32, 64)
        self.layer4 = self.conv_module(64, 128)
        self.layer5 = self.conv_module(128, 256)
        self.gap = self.global_avg_pool(256, num_classes)

    def forward(self, x):
        out1 = self.layer1(x)
        out = self.layer2(out1)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.gap(out)
        out = out.view(-1, num_classes)
        return out

    def conv_module(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
    def global_avg_pool(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))
    
if not (train_data_set.num_classes == test_data_set.num_classes):
    print("error: Numbers of class in training set and test set are not equal")
    exit()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_classes = train_data_set.num_classes
custom_model = CustomConvNet(num_classes=num_classes)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(custom_model.parameters(), lr=hyper_param_learning_rate)
for e in range(hyper_param_epoch):
    for i_batch, item in enumerate(train_loader):
        images = item['image']
        labels = item['label']
        # Forward pass
        outputs = custom_model(images)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i_batch + 1) % hyper_param_batch == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(e + 1, hyper_param_epoch, loss.item()))
            

# Test the model
custom_model.eval() 
with torch.no_grad():
    correct = 0
    total = 0
    i=0
    for item in test_loader:
        images = item['image']
        labels = item['label']
        outputs = custom_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += len(labels)
        correct += (predicted == labels).sum().item()
        output11 = outputs.cpu()
        output11 = output11.detach()
        output11 = output11.numpy()
        label = labels.cpu()
        label = label.detach()
        label = label.numpy()
        print(label)
        if i == 0 :
            output = output11
            path1 = label
        else:
            output = np.concatenate((output,output11))
            path1 =np.concatenate((path1,label))
        i+=1
        print(i)
    print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))
    
img_data=pd.concat([pd.DataFrame(path1,columns=['path']),pd.DataFrame(output)],axis=1)
img_data=img_data.drop_duplicates(subset=['path'])
img_data=img_data.set_index(np.arange(len(img_data)))
img_data.to_csv("cnn_1m.csv",encoding='utf-8-sig')

