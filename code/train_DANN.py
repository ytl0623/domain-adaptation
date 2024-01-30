import os
from torch.utils.data import Dataset
from pathlib import Path
import csv
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import cv2 as cv
from natsort import natsorted
import random
import torch

import torch.nn as nn
import time
import copy
from torch.autograd import Variable
import torchvision.models as models
import albumentations as Album
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torch.backends import cudnn

import torchvision
from torchvision import transforms
from torchvision.models import alexnet

from PIL import Image
from tqdm import tqdm

from models.models import *
from utils.utils import *

CUDA_DEVICES = 0

BATCH_SIZE = 3  #1977%3=0, 660%3=0

Train_Root = "/home/ytl0623/data/MI_proj3/data/DWI_PNG_train_bias/"
Val_Root = "/home/ytl0623/data/MI_proj3/data/ADC_PNG_val_bias/"
Train_Label_Root = "/home/ytl0623/data/MI_proj3/data/train_data.csv"
Val_Label_Root = '/home/ytl0623/data/MI_proj3/data/val_data.csv'

# Initial learning rate
init_lr = 0.01

# Training epochs
num_epochs = 30 #20改3

saved_model_path = "/home/ytl0623/data/MI_proj3/exp/DANN/" + str(init_lr) + "-" + str(num_epochs) + "/"

if not os.path.exists( saved_model_path ):
   os.makedirs( saved_model_path )

np.random.seed(2)
random.seed(2)
torch.manual_seed(2)
torch.cuda.manual_seed_all(2)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class IMAGE_Dataset(Dataset):
    def __init__(self,root_dir, label_root, transfrom = None):
        self.root_dir = Path(root_dir)
        self.labels = []
        self.transfrom = transfrom
        self.label_root = Path(label_root)
        self.patient = []
        
        patient_path = natsorted(os.listdir(root_dir))
        for i in patient_path:
          self.patient.append(os.path.join(root_dir,i))
        print(len(self.patient))

        label_df = pd.read_csv(label_root, encoding= 'unicode_escape')
        for j in range(len(self.patient)):
            label_df1 = np.array(label_df.poor_3m[label_df["PseudoNo"] == int(patient_path[j])])
            for i in range(len(label_df1)):
                self.labels.append(label_df1[i])
        print(len(self.labels))     

    def __len__(self):
        return len(self.patient)

    def __getitem__(self, index):
        patients = natsorted(os.listdir(self.patient[index]))
        start = int((len(patients)-18)/2)
        img_first = cv.imread(os.path.join(self.root_dir,self.patient[index],patients[start]))
        img_first = cv.cvtColor(img_first,cv.COLOR_BGR2GRAY)
        if self.transfrom is not None:
            img_first = self.transfrom(image = img_first)["image"]
        img_first = np.array(img_first)
        img_first = torch.from_numpy(img_first)
        img_first = img_first.float().div(255)
        img_first = img_first.unsqueeze(0)
        
        for i in range(start+1,start+18):
          img1 = cv.imread(os.path.join(self.root_dir,self.patient[index],patients[i]))
          img1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
          if self.transfrom is not None:
            img1 = self.transfrom(image = img1)["image"]
          img1 = np.array(img1)
          img1 = torch.from_numpy(img1)
          img1 = img1.float().div(255)
          img1 = img1.unsqueeze(0)
          img_first = torch.cat((img_first,img1),0)

        label = self.labels[index]

        return img_first, label

# Setting learning rate operation
def adjust_lr(optimizer, epoch):
    # 1/10 learning rate every 5 epochs
    lr = init_lr * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train():
    bestvalacc=0
    
    train_transform = Album.Compose([                                                                                  
        Album.Resize(224,224),
        Album.Normalize(mean=0,std=1)                
    ])
        
    train_set = IMAGE_Dataset(Path(Train_Root), Path(Train_Label_Root), train_transform)
    data_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_set = IMAGE_Dataset(Path(Val_Root), Path(Val_Label_Root), train_transform)
    val_data_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    classes = ["0","1"]
    classes.sort()
    classes.sort(key = len)
    
    # Loading model 
    model = dann_net(pretrained=True)  #AlexNet
    #print(net)
    
    model.features[0] = nn.Conv2d(18, 64, kernel_size=11, stride=4, padding=2)
    model.classifier[6] = nn.Linear(4096, 2)
    model.GD[6] = nn.Linear(4096, 2)
    #print(net)

    print("==========")

    model = model.cuda(CUDA_DEVICES)

    model.train()

    best_model_params = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    best_val_model_params = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    criterion = nn.CrossEntropyLoss()
    
    # Optimizer setting
    optimizer = torch.optim.SGD(params=model.parameters(), lr=init_lr, momentum=0.9)
      
    loss_class_list = []
    loss_target_list = []
    loss_source_list = []

    # Log 
    with open(saved_model_path + 'TrainingAccuracy.txt','w') as fAcc:
        print('Accuracy\n', file = fAcc)
        
    with open(saved_model_path + 'TrainingLoss.txt','w') as fLoss:
        print('Loss\n', file = fLoss)

    for epoch in range(num_epochs):
        model.train()
        localtime = time.asctime( time.localtime(time.time()) )
        
        print('Epoch: {}/{} --- < Starting Time : {} >'.format(epoch + 1,num_epochs,localtime))
        print('-' * len('Epoch: {}/{} --- < Starting Time : {} >'.format(epoch + 1,num_epochs,localtime)))

        training_loss = 0.0
        training_corrects = 0
        adjust_lr(optimizer, epoch)

        for i, (inputs, labels) in enumerate(tqdm(data_loader)):

            inputs = Variable(inputs.cuda(CUDA_DEVICES))
            labels = Variable(labels.cuda(CUDA_DEVICES))
            
            optimizer.zero_grad()

            #outputs = model(inputs)
           
            # STEP 1: train the classifier
            outputs = model(inputs)
            outputs_2 = outputs
                      
            loss_class = criterion(outputs, labels)  
            loss_class_list.append(loss_class.item())                
            loss_class.backward()  # backward pass: computes gradients
            
            # Load target batch
            target_images, target_labels = next(iter(val_data_loader))
            target_images = Variable(target_images.cuda(CUDA_DEVICES))
            target_labels = Variable(target_labels.cuda(CUDA_DEVICES))
            
            # STEP 2: train the discriminator: forward SOURCE data to Gd          
            outputs = model.forward(inputs, alpha=0.1)
            
            # source's label is 0 for all data    
            labels_discr_source = torch.zeros(BATCH_SIZE, dtype=torch.int64).cuda(CUDA_DEVICES)
            loss_discr_source = criterion(outputs, labels_discr_source)  
            loss_source_list.append(loss_discr_source.item())
            loss_discr_source.backward()
            
            # STEP 3: train the discriminator: forward TARGET to Gd          
            outputs = model.forward(target_images, alpha=0.1)           
            labels_discr_target = torch.ones(BATCH_SIZE, dtype=torch.int64).cuda(CUDA_DEVICES) # target's label is 1
            loss_discr_target = criterion(outputs, labels_discr_target)    
            loss_target_list.append(loss_discr_target.item())
            loss_discr_target.backward()    #update gradients 
            
            _, preds = torch.max(outputs_2.data, 1)

            #loss = criterion(outputs, labels)
            #loss.backward()
            
            loss = loss_class
            
            optimizer.step()

            training_loss += float(loss.item() * inputs.size(0))
            training_corrects += torch.sum(preds == labels.data)
            
        training_loss = training_loss / len(train_set)
        training_acc = training_corrects.double() /len(train_set)
        print('Training loss: {:.4f}\taccuracy: {:.4f}\n'.format(training_loss,training_acc))

        # Check best accuracy model ( but not the best on test )
        if training_acc > best_acc:
            best_acc = training_acc
            best_model_params = copy.deepcopy(model.state_dict())


        with open(saved_model_path + 'TrainingAccuracy.txt','a') as fAcc:
            print('{:.4f} '.format(training_acc), file = fAcc)
            
        with open(saved_model_path + 'TrainingLoss.txt','a') as fLoss:
            print('{:.4f} '.format(training_loss), file = fLoss)

        model = model.cuda(CUDA_DEVICES)
        model.eval()
        total_correct = 0
        total = 0
        class_correct = list(0. for i in enumerate(classes))
        class_total = list(0. for i in enumerate(classes))

        with torch.no_grad():
            for inputs, labels in tqdm(val_data_loader):
                inputs = Variable(inputs.cuda(CUDA_DEVICES))
                labels = Variable(labels.cuda(CUDA_DEVICES))

                outputs = model(inputs)
                
                _, predicted = torch.max(outputs.data, 1)
                
                # totoal
                total += labels.size(0)
                total_correct += (predicted == labels).sum().item()
                c = (predicted == labels).squeeze()
                
                # batch size
                for i in range(labels.size(0)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

            for i, c in enumerate(classes):
                print('Accuracy of %5s : %8.4f %%' % (
                c, 100 * class_correct[i] / class_total[i]))

            # Accuracy
            print('\nAccuracy on the ALL test images: %.4f %%' % (100 * total_correct / total))
            
            val_acc = 100 * total_correct / total
            
        print('val accuracy: {:.4f}\n'.format(val_acc))
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_model_params = copy.deepcopy(model.state_dict())
            torch.save(model, saved_model_path+'seq-resnet50-model-{:.2f}-best_val_acc.pth'.format(val_acc))
            
    total = sum([param.nelement() for param in model.parameters()])
    
    print("Number of parameter: %.2fM" % (total/1e6))
    
    # Save best training/valid accuracy model ( not the best on test )
    model.load_state_dict(best_model_params)
    best_model_name = saved_model_path+'seq-resnet50-model-{:.2f}-best_train_acc.pth'.format(best_acc)
    torch.save(model, best_model_name)
    print("Best model name : " + best_model_name)
    
if __name__ == '__main__':
    train()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
