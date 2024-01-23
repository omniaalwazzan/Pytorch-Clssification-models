from PIL import Image
import numpy as np
import shap
import torch.nn as nn
from torchvision import models
from torchvision.models import vgg16_bn, VGG16_BN_Weights, convnext_base, ConvNeXt_Base_Weights
import torch
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import shap
#%%
class convNext(nn.Module):
    def __init__(self):
        super(convNext, self).__init__()
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        feature = self.feature_extractor(x)
        return feature
#%%

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

#%%

model = convNext()
model.feature_extractor.register_forward_hook(get_activation('feature_extractor'))
x = torch.randn(1, 3, 512, 512)
output = model(x)
print(activation['feature_extractor'])
