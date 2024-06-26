# -*- coding: utf-8 -*-
"""

Author: Mason Lovejoy-Johnson
Contact Email: malovej2@ncsu.edu
Start Date: 3/1/2022
Last Update: 3/16/2022

This program will take the previously developed model and then apply it to a website
in order to allow for online use. This will use flask to connect with the html
required for website development.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import flask
from flask import Flask, request, render_template

app = Flask(__name__)

class_filenames = ['baki', 'berserk', 'bleach', 'eyeshield', 'fairytail', 'foodwars', 'gantz','gintama',
                   'haikyu', 'hajime', 'hxh', 'inuyasha', 'jojo', 'kingdom', 'naruto', 'onepiece', 'pokemon',
                   'slamdunk', 'vagabond']

class_index = ['Baki', 'Berserk', 'Bleach', 'Eyeshield 21', 'Fairy Tail','Food Wars!: Shokugeki no Soma',
               'Gantz', 'Gintama', 'Haikyu!', 'Hajime no Ippo', 'Hunter x Hunter', 'Inuyasha','Jojo\'s Bizzare Adventure',
               'Kingdom', 'Naruto', 'One Piece', 'Pokémon Adventures', 'Slam Dunk', 'Vagabond' ]

image_list = []
image_file = []
for image_idx in range(0, len(class_index)):
    image_file.append('file path' +str(class_filenames[image_idx])+'_cover.jpg')
    image_list.append(Image.open('file path' +str(class_filenames[image_idx])+'_cover.jpg'))
    

def transform_image(image_bytes):
    #converting the image to a numpy array
    image_array = np.asarray(image_bytes)
    #converting the numpy array to a pytorch tensor
    image_tensor = torch.unsqueeze(torch.from_numpy(image_array).float(), 0)
    image_tensor = torch.unsqueeze(image_tensor, 0)
    return image_tensor

#gets the prediction of the model and output its guess as an int for the index of the manga
def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = int(y_hat.item())
    return predicted_idx

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

#loading in the already trained model and setting it to evaluation mode
model.load_state_dict(torch.load('file path'))
model.eval()


@app.route("/")
#initializing the website
def index():
    return flask.render_template('cnn_website.html', label="Predict")

@app.route("/home")
#function that allows the user to go home after clicking on a button
def send_home():
    return flask.render_template('cnn_website.html', label="Predict")

@app.route('/predict', methods=['POST'])
       
def make_predict():
    #makes a prediction for a manga after being provided with an image.
    if request.method == 'POST':
        file = request.files['image']
        
        if not file: return render_template('cnn_website.html', label="No file")
        
        img_bytes = Image.open(file).convert('L').resize((70, 110))
        label_idx = get_prediction(image_bytes=img_bytes)
        
        #shows a new page that shows the prediction of the model
        return render_template('cnn_website_after.html', label=class_index[label_idx], filename=class_filenames[label_idx] +'_cover.jpg')

if __name__ == '__main__':
    
    app.run()