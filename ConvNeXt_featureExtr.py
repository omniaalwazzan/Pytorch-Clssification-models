# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 15:19:01 2023

@author: Omnia
"""

import torchvision
from torchvision import models
from torch import nn
import torch
from torchsummary import summary

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class convNext(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        convNext = models.convnext_base(pretrained=True)
        feature_extractor = nn.Sequential(*list(convNext.children())[:-1])
        self.feature = feature_extractor
        self.calssifier =nn.Sequential(nn.Flatten(1, -1),
                                       nn.Dropout(p=0.2),
                                       nn.Linear(in_features=1024, out_features=3))

    def forward(self, x):
        feature = self.feature(x) # this feature we can use when doing stnad.Att
        flatten_featur = feature.view(feature.size(0), -1) #this we need to plot tsne
        x = self.calssifier(feature)
        return flatten_featur, x
    
model =convNext()
summary(model, (3,224,224))
img = torch.randn(2,3,224,224)
fea,out = model(img)
print(f"shape of feature:{fea.shape}\nshape of output {out.shape}")
