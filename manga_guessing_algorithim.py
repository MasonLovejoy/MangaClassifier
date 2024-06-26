# -*- coding: utf-8 -*-
"""

Author: Mason Lovejoy-Johnson
Contact Email: malovej2@ncsu.edu
Start Date: 1/12/2022
Last Update: 3/16/2022

The Goal of this program is to predict the manga given a single panel
using a convolutional neural network (CNN). The input is a 3-d tensor
of 2-d greyscale images and it will output a proability of what manga the 
neural network thinks the panel is from taking the highest probability.
"""

import matplotlib.pyplot as plt
import torch
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F

def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    run()
    
#loading in previously created tensors
image_list = torch.load('file path to the downloaded image tensors')
targets = torch.load('file path to the target tensor')
image_count = len(image_list[:,0,0])
tensor_list = torch.unsqueeze(image_list, 1)

val_size = int(0.10 * image_count)
train_size = image_count - val_size

class DataSet:

    def __init__(self, data, label_data):
        self.images = data
        self.labels = label_data

    def __len__(self):
        #This finds how many points are in our dataset
        return len(self.images)

    def __getitem__(self, idx):
        # This allows for pytorch to iterate throught the dataset
        # in the pytorch data loader by calling for specific index
    
        img = self.images[idx]
        label = self.labels[idx]

        return img, label

#setting the seed for pytorch to generate random numbers
random_seed = 42
torch.manual_seed(random_seed)

#putting our tensors into the DataSet class
data = DataSet(data = tensor_list, label_data = targets)

#splitting them into a training and validation set
train_ds, val_ds = random_split(data, [train_size, val_size])

#sending the seprate datasets to the dataloader
batch_size = 8
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers = 4, pin_memory = True)


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
   
#initializing the model
model = manga_cnn_model()    

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

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        #returns the batch of data after moving it to the new device
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        #number of batches
        return len(self.dl)   
  
#moving the data to the gpu
device = get_default_device()
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)

model = to_device(manga_cnn_model(), device)

torch.no_grad()

if __name__ == '__main__':         
    def evaluate(model, val_loader):
        model.eval()
        outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)
    
    def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
        history = []
        optimizer = opt_func(model.parameters(), lr)
        for epoch in range(epochs):
            # Training Phase 
            model.train()
            train_losses = []
            for batch in train_loader:
                loss = model.training_step(batch)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            # Validation phase
            result = evaluate(model, val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            model.epoch_end(epoch, result)
            history.append(result)
        return history
         
    #setting hyperparamters and optimization function
    num_epochs = 12
    opt_func = torch.optim.Adam
    lr = 0.00001

    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)               
        
    def plot_stats(history):
        #plotting the results of the model after the training phase
        accuracies = [x['val_acc'] for x in history]
        train_losses = [x.get('train_loss') for x in history]
        val_losses = [x['val_loss'] for x in history]
        plt.plot(train_losses, '-bx')
        plt.plot(val_losses, '-rx')
        plt.plot(accuracies, '-gx')
        plt.xlabel('epoch')
        plt.ylabel('loss and accuracy')
        plt.legend(['Training', 'Validation', 'Accuracy'])
        plt.title('Loss and Accuracy vs. No. of epochs for 20 labels');            
                
    plot_stats(history)            
            
            
            
            
            
            
            
            
            