# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 22:16:50 2022

@author: Omnia
"""

import timm
import torch
from torchsummary import summary 

def load_model():

    model =  timm.create_model('convnext_base', pretrained=True,num_classes=3) 


    # Disable gradients on all model parameters to freeze the weights
    for param in model.parameters():
        param.requires_grad = False

    for param in model.head.parameters():
        param.requires_grad = False

    # Unfreeze the last stage
    for param in model.stages[3].parameters():
        param.requires_grad = True
    
    return model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_3= load_model()
model_3 = model_3.to(device=DEVICE,dtype=torch.float)
print(summary(model_3,(3, 224, 224)))
