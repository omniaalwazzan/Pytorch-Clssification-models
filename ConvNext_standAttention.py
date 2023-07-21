# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 20:55:01 2023

@author: Omnia
"""


import torch.nn as nn
import math as m
import torch
import torch.nn.functional as F
#from torchsummary import summary
from torchvision import models

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

h_dim = 1
class MultiHeadAttention(nn.Module):
    # default values for the diminssion of the model is 8 and heads 4
    def __init__(self, d_model=8, num_heads=4, dropout=0.1):
        super().__init__()

        # d_q, d_k, d_v
        self.d = d_model//num_heads

        self.d_model = d_model
        self.num_heads = num_heads

        self.dropout = nn.Dropout(dropout)

        ##create a list of layers for K, and a list of layers for V
        
        self.linear_Qs = nn.ModuleList([nn.Linear(d_model, self.d)
                                        for _ in range(num_heads)])
        self.linear_Ks = nn.ModuleList([nn.Linear(d_model, self.d)
                                        for _ in range(num_heads)])
        self.linear_Vs = nn.ModuleList([nn.Linear(d_model, self.d)
                                        for _ in range(num_heads)])

        self.mha_linear = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V):
        # shape(Q) = [B x feature_dim x D/num_heads] = [B x feature_dim x d_k]
        # shape(K, V) = [B x feature_dim x d_k]

        Q_K_matmul = torch.matmul(Q, K.permute(0, 2, 1))
        scores = Q_K_matmul/m.sqrt(self.d)
        # shape(scores) = [B x feature_dim x feature_dim]

        attention_weights = F.softmax(scores, dim=-1)
        # shape(attention_weights) = [B x feature_dim x feature_dim]

        output = torch.matmul(attention_weights, V)
        # shape(output) = [B x feature_dim x D/num_heads]

        return output, attention_weights

    def forward(self, x):
        # shape(x) = [B x feature_dim x D]

        Q = [linear_Q(x) for linear_Q in self.linear_Qs]
        #print('shape of Query',Q[0].shape)
        K = [linear_K(x) for linear_K in self.linear_Ks]
        #print('shape of Key',K[0].shape)        
        V = [linear_V(x) for linear_V in self.linear_Vs]
        #print('shape of Value',V[0].shape)

        # shape(Q, K, V) = [B x feature_dim x D/num_heads] * num_heads

        output_per_head = []
        attn_weights_per_head = []
        # shape(output_per_head) = [B x feature_dim x D/num_heads] * num_heads
        # shape(attn_weights_per_head) = [B x feature_dim x feature_dim] * num_heads
        for Q_, K_, V_ in zip(Q, K, V):
            
            ##run scaled_dot_product_attention
            output, attn_weight = self.scaled_dot_product_attention(Q_, K_, V_)

            # shape(output) = [B x feature_dim x D/num_heads]
            # shape(attn_weights_per_head) = [B x feature_dim x feature_dim]
            output_per_head.append(output)
            attn_weights_per_head.append(attn_weight)
        #print('shape of attnention weights',attn_weight[0].shape)

        output = torch.cat(output_per_head, -1)
        attn_weights = torch.stack(attn_weights_per_head).permute(1, 0, 2, 3)
        # shape(output) = [B x feature_dim x D]
        # shape(attn_weights) = [B x num_heads x feature_dim x feature_dim]
        
        projection = self.dropout(self.mha_linear(output))

        return projection#, attn_weights


class convNext(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        convNext = models.convnext_base(pretrained=True)
        feature_extractor = nn.Sequential(*list(convNext.children())[:-1])
        self.feature = feature_extractor
        self.attention = MultiHeadAttention(d_model=h_dim , num_heads=h_dim)
        self.calssifier =nn.Sequential(nn.Flatten(1, -1),
                                       nn.Dropout(p=0.2),
                                       nn.Linear(in_features=1024, out_features=3))

    def forward(self, x):
        feature = self.feature(x) # this feature we can use when doing stnad.Att
        flatten_featur = feature.view(feature.size(0), -1,feature.size(2) *feature.size(3)) # this for attention (bs,channel dim, hxw)
        atten_feature = self.attention(flatten_featur) # shape [bs, 1024, 1]
        #print(f'featur:{atten_feature.shape}')
        x = self.calssifier(atten_feature)
        #return feature,flatten_featur, x
        return atten_feature,x

    


    
# cnn_m =convNext().to(device=DEVICE,dtype=torch.float32)
# summary(cnn_m, (3,224,224))
# img = torch.randn(2,3,224,224).to(device=DEVICE,dtype=torch.float32)
# feature,out= cnn_m(img)
#print(f"shape of feature:{feature.shape}\nshape of output {out.shape}")
