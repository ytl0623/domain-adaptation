#!/usr/bin/env python
# coding: utf-8

# **Import libraries**

# In[1]:


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


# **Set Arguments**

# In[2]:


DEVICE = 'cuda:1'      # 'cuda' or 'cpu'

NUM_CLASSES = 2
DATASETS_NAMES = ['ADC', 'DWI']
CLASSES_NAMES = ['0', '1']

# Hyperparameters
MOMENTUM = 0.9        # Hyperparameter for SGD, keep this at 0.9 when using SGD
WEIGHT_DECAY = 5e-5   # Regularization, you can keep this at the default
GAMMA = 0.1           # Multiplicative factor for learning rate step-down

# Hyperparameters for grid search
BATCH_SIZE = 659
LR = 1e-3             # The initial Learning Rate
NUM_EPOCHS = 30       # Total number of training epochs (iterations over dataset)
STEP_SIZE = 20        # How many epochs before decreasing learning rate (if using a step-down policy)
ALPHA = 0.1           # alpha
ALPHA_EXP = False

EVAL_ACCURACY_ON_TRAINING = False
SHOW_IMG = True      # if 'True' show images and graphs on output


# **Define Data Preprocessing**

# In[3]:


transf = Album.Compose([
  Album.Resize(224,224),
  Album.Normalize(mean=0,std=1)
])


# **Prepare Dataset**

# In[4]:


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
        
        #print(len(self.patient))

        label_df = pd.read_csv(label_root, encoding= 'unicode_escape')
        for j in range(len(self.patient)):
            label_df1 = np.array(label_df.poor_3m[label_df["PseudoNo"] == int(patient_path[j])])
            for i in range(len(label_df1)):
                self.labels.append(label_df1[i])
                
        #print(len(self.labels))     

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


# In[5]:


# Define datasets root
Train_Root = "/home/ytl0623/data/MI_proj3/data/DWI_PNG_train_bias/"
Val_Root = "/home/ytl0623/data/MI_proj3/data/ADC_PNG_val_bias/"

Train_Label_Root = "/home/ytl0623/data/MI_proj3/data/train_data.csv"
Val_Label_Root = '/home/ytl0623/data/MI_proj3/data/val_data.csv'

ADC_dataset = IMAGE_Dataset(Path(Val_Root), Path(Val_Label_Root), transf)
DWI_dataset = IMAGE_Dataset(Path(Train_Root), Path(Train_Label_Root), transf)

# Check dataset sizes
# print(f"ADC Dataset: {len(ADC_dataset)}")
# print(f"DWI Dataset: {len(DWI_dataset)}")


# **Prepare Dataloaders**

# In[6]:


# Dataloaders iterate over pytorch datasets and transparently provide useful functions (e.g. parallelization and shuffling)
ADC_dataloader = DataLoader(dataset=ADC_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
DWI_dataloader = DataLoader(dataset=DWI_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# **Prepare Network for training**

# In[10]:


cudnn.benchmark # Calling this optimizes runtime

USE_DOMAIN_ADAPTATION = True 
USE_VALIDATION = False
transfer_set = "ADC"

source_dataloader = DWI_dataloader
test_dataloader = ADC_dataloader
target_dataloader = ADC_dataloader

print(len(source_dataloader), len(target_dataloader))

# Loading model 
net = dann_net(pretrained=True).to(DEVICE)  #AlexNet
#print(net)

net.features[0] = nn.Conv2d(18, 64, kernel_size=11, stride=4, padding=2).to(DEVICE)
net.classifier[6] = nn.Linear(4096, 2).to(DEVICE)
net.GD[6] = nn.Linear(4096, 2).to(DEVICE)
#print(net)

# Define loss function: CrossEntrpy for classification
criterion = nn.CrossEntropyLoss()

# Choose parameters to optimize
parameters_to_optimize = net.parameters() # In this case we optimize over all the parameters of AlexNet

# Define optimizer: updates the weights based on loss (SDG with momentum)
optimizer = optim.SGD(parameters_to_optimize, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

# Define scheduler -> step-down policy which multiplies learning rate by gamma every STEP_SIZE epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

if ALPHA_EXP : 
  # ALPHA exponential decaying as described in the paper
  p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
  ALPHA = 2. / (1. + np.exp(-10 * p)) - 1


# **Train**

# In[11]:


current_step = 0
accuracies_train = []
accuracies_validation = []
loss_class_list = []
loss_target_list = []
loss_source_list = []

# Start iterating over the epochs
for epoch in range(NUM_EPOCHS):
  
  net.train(True)

  print(f"--- Epoch {epoch+1}/{NUM_EPOCHS}, LR = {scheduler.get_last_lr()}")
  
  # Iterate over the dataset
  gg = 0
  for source_images, source_labels in source_dataloader:
    print(gg)
    gg+=1
    source_images = source_images.to(DEVICE)
    source_labels = source_labels.to(DEVICE)    

    optimizer.zero_grad() # Zero-ing the gradients
    
    # STEP 1: train the classifier
    outputs = net(source_images)          
    loss_class = criterion(outputs, source_labels)  
    loss_class_list.append(loss_class.item())                
    loss_class.backward()  # backward pass: computes gradients

    # Domain Adaptation (Cross Domain Validation)
    if USE_DOMAIN_ADAPTATION :

      # Load target batch
      target_images, target_labels = next(iter(target_dataloader))
      target_images = target_images.to(DEVICE)

      # if ALPHA_EXP : 
      #   # ALPHA exponential decaying as described in the paper
      #   p = float(i + epoch * len_dataloader) / NUM_EPOCHS / len_dataloader
      #   ALPHA = 2. / (1. + np.exp(-10 * p)) - 1

      # STEP 2: train the discriminator: forward SOURCE data to Gd          
      outputs = net.forward(source_images, alpha=ALPHA)
    
      # source's label is 0 for all data    
      labels_discr_source = torch.zeros(BATCH_SIZE, dtype=torch.int64).to(DEVICE)
      #print(labels_discr_source)
      loss_discr_source = criterion(outputs, labels_discr_source)  
      loss_source_list.append(loss_discr_source.item())
      loss_discr_source.backward()

      # STEP 3: train the discriminator: forward TARGET to Gd          
      outputs = net.forward(target_images, alpha=ALPHA)           
      labels_discr_target = torch.ones(BATCH_SIZE, dtype=torch.int64).to(DEVICE) # target's label is 1
      loss_discr_target = criterion(outputs, labels_discr_target)    
      loss_target_list.append(loss_discr_target.item())
      loss_discr_target.backward()    #update gradients 

    optimizer.step() # update weights based on accumulated gradients
    #print(loss_discr_source, loss_discr_target)          
    
  # --- Accuracy on training
  if EVAL_ACCURACY_ON_TRAINING:
    with torch.no_grad():
      net.train(False)

      running_corrects_train = 0

      for images_train, labels_train in source_dataloader:
        # images, labels = next(iter(source_dataloader))
        images_train = images_train.to(DEVICE)
        labels_train = labels_train.to(DEVICE)

        # Forward Pass
        outputs_train = net(images_train)

        # Get predictions
        _, preds = torch.max(outputs_train.data, 1)

        # Update Corrects
        running_corrects_train += torch.sum(preds == labels_train.data).data.item()

    # Calculate Accuracy
    accuracy_train = running_corrects_train / float(len(source_dataloader)*(target_dataloader.batch_size))
    accuracies_train.append(accuracy_train)
    print('Accuracy on train (DWI):', accuracy_train)
    
  # --- VALIDATION SET
  if USE_VALIDATION : 
    # now train is finished, evaluate the model on the target dataset 
    net.train(False) # Set Network to evaluation mode
      
    running_corrects = 0
    for images, labels in target_dataloader:
      images = images.to(DEVICE)
      labels = labels.to(DEVICE)
      
      outputs = net(images)
      _, preds = torch.max(outputs.data, 1)
      running_corrects += torch.sum(preds == labels.data).data.item()

    # Calculate Accuracy
    accuracy = running_corrects / float( len(target_dataloader)*(target_dataloader.batch_size) )
    accuracies_validation.append(accuracy)
    print(f"Accuracy on validation ({transfer_set}): {accuracy}")

  # Step the scheduler
  current_step += 1
  scheduler.step()


# **Test**

# In[ ]:


net = net.to(DEVICE) # this will bring the network to GPU if DEVICE is cuda
net.train(False) # Set Network to evaluation mode

running_corrects = 0
for images, labels in tqdm(test_dataloader):
  images = images.to(DEVICE)
  labels = labels.to(DEVICE)

  # Forward Pass
  outputs = net(images)

  # Get predictions
  _, preds = torch.max(outputs.data, 1)

  # Update Corrects
  running_corrects += torch.sum(preds == labels.data).data.item()

# Calculate Accuracy
accuracy = running_corrects / float(len(ADC_dataset))

print('\nTest Accuracy (ADC): {} ({} / {})'.format(accuracy, running_corrects, len(ADC_dataset)))


# In[ ]:


if USE_VALIDATION : 
  print(f"Validation on:  {transfer_set}")
  print(f"accuracy_valid: {accuracies_validation[-1]:.4f}")
  print(accuracies_validation)
    
print(f"Test accuracy:  {accuracy:.4f}")
print(f"Val on {transfer_set}, LR = {LR}, ALPHA = {ALPHA}, BATCH_SIZE = {BATCH_SIZE}")


# In[ ]:


if USE_DOMAIN_ADAPTATION :
  # Plot losses 
  plotLosses(loss_class_list, loss_source_list, loss_target_list, n_epochs=len(loss_class_list), show=SHOW_IMG)

