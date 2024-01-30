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

CUDA_DEVICES = 0

Test_Root = '/content/DWI_PNG_val_bias'
Test_Label_Root = '/content/val_data.csv'
PATH_TO_WEIGHTS = '/content/drive/My Drive/seq-resnet50-model-46.06-best_val_acc.pth'  # Your model name

def test():
    data_transform = Album.Compose([                           
        Album.Resize(224,224),
        Album.Normalize(mean=0,std=1),                                             
    ])
    test_set = IMAGE_Dataset(Path(Test_Root), Path(Test_Label_Root), data_transform)
    data_loader = DataLoader(dataset=test_set, batch_size=16, shuffle=False, num_workers=0)
    classes = ["0","1"]
    classes.sort()

    # Load model
    model = torch.load(PATH_TO_WEIGHTS)

    model.eval()

    total_correct = 0
    total_false = 0
    total = 0
    class_correct = list(0. for i in enumerate(classes))
    class_total = list(0. for i in enumerate(classes))

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader):
            inputs = Variable(inputs.cuda(CUDA_DEVICES))
            labels = Variable(labels.cuda(CUDA_DEVICES))
            
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)

            # totoal
            total += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            total_false += (predicted != labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                  label = labels[i]
                  class_correct[label] += c[i].item()
                  class_total[label] += 1
           
    for i, c in enumerate(classes):
        print('Accuracy of %5s : %8.4f %%' % (c, 100 * class_correct[i] / class_total[i]))
        
    # Accuracy
    print('\nAccuracy on the ALL test images: %.4f %%' % (100 * total_correct / total))

if __name__ == "__main__":
    test()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    