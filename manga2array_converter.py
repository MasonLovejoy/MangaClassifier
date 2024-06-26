# -*- coding: utf-8 -*-
"""

Author: Mason Lovejoy-Johnson
Contact Email: malovej2@ncsu.edu
Start Date: 1/12/2022
Last Update: 3/16/2022

This part of the program loads in image files from the internet and converts them
into a matrix of 3 dimensions. This file is then saved into a .npy file which is
just keeping the file as an array so it can be called easily back in the future.
"""

import numpy as np
import os
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
dir_name = 'file path to the directory of manga panel'

manga_folders = os.listdir(dir_name)

#creating a constant for image size and two empty lists for storage
image_size  = (70, 110)
image_list  = []
image_label = []

#creating and array for indexing the panel to the correct manga
manga_idx = np.linspace( 0, 19, 20, dtype=int)

for folder_idx in range(0, len(manga_folders)):
    
    #loads each folder of manga in and the sperates them into individual panels
    print('Currently turning', str(manga_folders[folder_idx]), 'into a pytorch tensor')
    
    manga_panels = os.listdir(dir_name + '\\' + str(manga_folders[folder_idx]))
    
    #loads the indivuidual panels in and converts them to a specific number
    #assigns a numeric index to each panel as a target for training.
    for panel_idx in range(0, len(manga_panels)):
        
        if panel_idx % 1000 == 0:
            
            print('Current Panel:', manga_panels[panel_idx],'with Panel index:', panel_idx)
            
        image = Image.open(dir_name + '\\' + str(manga_folders[folder_idx]) + '\\' + manga_panels[panel_idx]
                           ).convert('L')

        if image.size[0] > image.size[1]:
            
            '''
            Some manga images are two panels saved and output as one larger panel.
            This if statement takes these images and seprates them into two images.
            Then adds both to the image list.
            '''
            
            half_width = int(image.size[0]/ 2)
    
            image_panel1 = image.crop(( 0, 0, half_width, image.size[1]))
            image_panel2 = image.crop(( half_width, 0, image.size[0], image.size[1]))

            image_list.append(np.asarray(image_panel1.resize((image_size))))
            image_label.append(manga_idx[folder_idx])
            image_list.append(np.asarray(image_panel2.resize((image_size))))
            image_label.append(manga_idx[folder_idx])
            
        else:
            image_list.append(np.asarray(image.resize((image_size))))
            image_label.append(manga_idx[folder_idx])
 
#converting the list of images and targets to a numpy array
image_array = np.asarray(image_list)
targets = np.asarray(image_label)

#converting the numpy array of images and targets to pytorch tensors
tensor_image = torch.from_numpy(image_array).float()
targets  =  torch.from_numpy(targets)
targets = targets.to(torch.int64)

#saving the tensors
torch.save(targets, 'targets_tensor.pt')
torch.save(tensor_image, 'image_tensor.pt')
      







