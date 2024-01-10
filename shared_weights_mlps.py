import torch
import torch.nn as nn
    
#%%

class Linear_Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Linear_ = nn.Sequential(
            nn.Linear(in_channels,out_channels),
            nn.LayerNorm(out_channels)
            )

    def forward(self, x):
        return self.Linear_(x)
#%%
# my model
class MyEnsemble(nn.Module):
    def __init__(self, cnv_in, dna_in, nb_classes):
        super(MyEnsemble, self).__init__()
        
        self.cnv = Linear_Layer(cnv_in,100)
        self.dna = Linear_Layer(dna_in,100)

        self.shared1 = Linear_Layer(100,100)
        self.shared2 = Linear_Layer(100, 500)
        self.shared3 = Linear_Layer(500, 256)
        self.dropout = nn.Dropout(p=0.1)
        

        # Create new classifier
        self.layer_out = nn.Linear(1000, nb_classes)
    
    def forward(self, cnv, dna):
        x1 = self.cnv(cnv)
        x2 = self.dna(dna)
        
        x1_sh1 = self.shared1(x1)
        x2_sh1 = self.shared1(x2)
        
        x1_sh2 = self.shared2(x1_sh1)
        x2_sh2 = self.shared2(x2_sh1)

        x = torch.cat((x1_sh2, x2_sh2), dim=1)
        x = x.flatten(start_dim=1)
        print(x.shape)
    
        x = self.layer_out(x)
        return x

#%%


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from torchsummary import summary
cnv_in = 29
dna = 8000
gt = 20
model = MyEnsemble(cnv_in,dna,gt)
model.to(device=DEVICE,dtype=torch.float)
summary(model, [(1,29),(1,8000)])
