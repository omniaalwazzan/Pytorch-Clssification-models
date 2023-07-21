
from torch import nn
from torch.nn import Parameter
import torch
from torch.utils.model_zoo import load_url as load_state_dict_from_url


model_urls = {
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'
}



class PathNet(nn.Module):

    def __init__(self, features, path_dim=64, act=None, num_classes=3):
        super(PathNet, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(1024, path_dim),
            nn.ReLU(True),
            nn.Dropout(0.05)
        )

        self.linear = nn.Linear(path_dim, num_classes)
        self.act = act

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

        #dfs_freeze(self.features)

    def forward(self,x):
        #x = kwargs['x_path']
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        features = self.classifier(x)
        hazard = self.linear(features)

        if self.act is not None:
            hazard = self.act(hazard)

            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift

        return  features,hazard
        #return  hazard



def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {

    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



def get_vgg(arch='vgg19_bn', cfg='E', act=None, batch_norm=True, label_dim=3, pretrained=True, progress=True):
    model = PathNet(make_layers(cfgs[cfg], batch_norm=batch_norm), act=act, num_classes=label_dim)
    
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls[arch], progress=progress)

        for key in list(pretrained_dict.keys()):
            if 'classifier' in key: pretrained_dict.pop(key)

        model.load_state_dict(pretrained_dict, strict=False)
        print("Initializing Path Weights")

    return model
