import pandas as pd
import numpy as np
import torch 
from torch import nn
from torch.nn import init, Parameter
from torchsummary import summary
import torch.nn.functional as F
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"




class MaxNet(nn.Module):
    def __init__(self, input_dim=3, omic_dim=32, dropout_rate=0.25):
        super(MaxNet, self).__init__()
        hidden = [64, 48, 32, 32]

        encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))
        
        encoder2 = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))
        
        encoder3 = nn.Sequential(
            nn.Linear(hidden[1], hidden[2]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder4 = nn.Sequential(
            nn.Linear(hidden[2], omic_dim),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))
        
        self.encoder = nn.Sequential(encoder1, encoder2, encoder3, encoder4)


    def forward(self, x):
        features = self.encoder(x)
        return features

#model = MaxNet()
#print(model)

def model_SNN() -> MaxNet:
    model = MaxNet()
    return model

modelA  = model_SNN()
#print(modelA)
summary(modelA,(3,),1,'cpu')
