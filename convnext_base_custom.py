
import torch
from torchvision import transforms
import pandas as pd
import os 
import PIL 
from PIL import Image 
import torch 
from torch import nn
from torchvision import models
from torchsummary import summary 
from torchvision import models

from torchvision.models import convnext_base, ConvNeXt_Base_Weights

#convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
#convNext = models.convnext_base(pretrained=True)
#model = models.convnext_base(pretrained=True)


class convNext(nn.Module):
    def __init__(self, n_classes=32):
        super().__init__()
        convNext = models.convnext_base(pretrained=True)
        convNext.avgpool = nn.AdaptiveAvgPool2d((1))
        convNext.classifier = nn.Sequential(nn.Flatten(1, -1),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1024, out_features=n_classes)
        )
        self.base_model = convNext

        #self.sigm = nn.Sigmoid()

    def forward(self, x):
        print(x.shape)

        x = self.base_model(x)
        print(x.shape)
        return x




cnn_net = convNext()
input_size = (3,224,224)
summary(cnn_net,input_size)
