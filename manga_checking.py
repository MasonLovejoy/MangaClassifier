# -*- coding: utf-8 -*-
"""

Author: Mason Lovejoy-Johnson
Contact Email: malovej2@ncsu.edu
Start Date: 2/24/2022
Last Update: 3/16/2022

The Goal of this program is to use our previously trained model to predict
if a manga panel is from one of the originally provided manga's. This will
also be used to output a plot of the accuracy of the network on a test set.
"""

import numpy as np
import torch
import matplotlib.pyplot as py
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import os

#loading images into from the test set
dir_name = 'file path to the manga panel test set directory'
manga_folders = os.listdir(dir_name)

#setting the image to the proper size
size  = (70, 110)

def to_tensor(path, image_size):
    #loading in the in a test image
    image = Image.open(path).convert('L').resize((image_size))
    #converting the image to a numpy array
    image_array = np.asarray(image)
    #converting the numpy array to a pytorch tensor
    image_tensor = torch.unsqueeze(torch.from_numpy(image_array).float(), 0)
    image_tensor = torch.unsqueeze(image_tensor, 0)
    return image_tensor

#class to find the loss and output the lossess and accuracies as the model trains
class ImageClassification(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print('Epoch ', epoch,', train_loss:', result['train_loss'], ', val_loss:',
              result['val_loss'], ', val_acc:', result['val_acc'])
        
#find the accuracy of a given model by suming the amount of correct classifications
#then dividing by the size of the prediction set
def accuracy(outputs, labels):
    _, preds =  torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

#creating the acutal convolutional neural network
class manga_cnn_model(ImageClassification):
    
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            
            nn.Conv2d(1, 100, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(100),
            nn.ReLU(inplace=True),
            nn.Conv2d(100, 200, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.1),
            nn.Conv2d(200, 300, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.1),
            nn.BatchNorm2d(300),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(300, 400, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.2),
            nn.Conv2d(400, 400, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(400),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(400, 600, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.3),
            nn.BatchNorm2d(600),
            nn.ReLU(inplace=True),
            nn.Conv2d(600, 600, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.3),
            nn.Conv2d(600, 600, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.3),
            nn.BatchNorm2d(600),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(600, 850, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.4),
            nn.BatchNorm2d(850),
            nn.ReLU(inplace=True),
            nn.Conv2d(850, 850, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.4),
            nn.Conv2d(850, 850, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.5),
            nn.BatchNorm2d(850),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(850, 900, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.5),
            nn.Conv2d(900, 900, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.5),
            nn.BatchNorm2d(900),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        
            nn.Flatten(),
            nn.Linear(5400, 2500),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(2500, 500),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(500, 19)   )                    
             
    def forward(self, xb):
        return self.network(xb)
    
def get_default_device():
    #allows for the use of a gpu if possible
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    #moving the data to the specified device
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

#moving the model to the gpu
device = get_default_device()
model = to_device(manga_cnn_model(), device)

#loading in the already trained model and setting it to evaluation mode
model.load_state_dict(torch.load('C:\\Manga_guesser\\final_model_dict.pt'))
model.eval()

def predict_image(img, model):
    xb = to_device(img, device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return preds[0].item()

def manga_acc(images, mangaid):
    #finding the accuracy of a set of images from a specific manga
    prediction_list = []
    correct_list = np.zeros(len(images))

    for prediction_idx in range(0,len(images)):
        prediction_list.append(predict_image(images[prediction_idx], model))
        if prediction_list[prediction_idx] == mangaid:
            correct_list[prediction_idx] = 1
    
    prediction_accuracy = np.sum(correct_list)/len(images)
    return prediction_accuracy, prediction_list

acc_list = []
ans_list = []

for m_id in range(0, len(manga_folders)):
    
    #loading in the manga test panels
    manga_testpanels = os.listdir(dir_name + '\\' + str(manga_folders[m_id]))

    image_list = []
    
    #finding the accuracy of each specific manga
    for image_idx in range(1, len(manga_testpanels)):
        image_list.append(to_tensor(dir_name + '\\' + str(manga_folders[m_id]) + '\\'+ str(manga_testpanels[image_idx]), size))
        
    
    acc_list.append(manga_acc(image_list, m_id)[0])
    ans_list.append(manga_acc(image_list, m_id)[1])
 
#computing overall accuracy of the model
average_acc = np.average(acc_list)

#plotting a chart of the accuracy for each manga
py.title('Precent Accuracy of Model')
py.xlabel('Manga Title')
py.ylabel('Accuracy (%)')
py.bar(manga_folders, acc_list, color ='blue')
py.hlines(average_acc,-1,20,colors='r', linestyles='dashed', label='Average Accuracy')
py.legend()
py.show()
    
    
    
    
    
    
    
    
    