
import torchvision
from torchvision import models
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

from torch import nn
import torch
from torchsummary import summary
from torchvision.models import ResNet18_Weights





DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cmmd_gt = 2
class Res(nn.Module):
    def __init__(self, n_classes=cmmd_gt):
        super().__init__()
        R18 = models.resnet18(pretrained=True)
        feature_extractor = nn.Sequential(*list(R18.children())[:-2])
        self.feature = feature_extractor
        self.calssifier =nn.Sequential(nn.Linear(512 * 16 * 16, 1024),
                                       #nn.ReLU(True),
                                       #nn.Dropout(0.25),
                                       nn.Linear(1024, 512),
                                       #nn.ReLU(True),
                                       nn.Dropout(0.25),
                                       nn.Linear(512, n_classes))

    def forward(self, x):
        feature = self.feature(x) # this feature we can use when doing stnad.Att
        print(feature.shape)
        flatten_featur = feature.reshape(feature.size(0), -1) #this we need to plot tsne
        print(flatten_featur.shape)
        #x =  x.view(x.size(0), -1)
        x = self.calssifier(flatten_featur)
        return flatten_featur, x
    
model =Res()
summary(model, (3,512,512))
img = torch.randn(2,3,512,512)
fea,out = model(img)
print(f"shape of feature:{fea.shape}\nshape of output {out.shape}")#
out.shape
