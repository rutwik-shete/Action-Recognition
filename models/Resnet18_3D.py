from transformers import AutoImageProcessor, TimesformerForVideoClassification
from torch import nn

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18

from Constants import CATEGORY_INDEX

import math
import numpy as np


class Resnet18_3d(nn.Module):
    def __init__(self, args, num_classes=len(CATEGORY_INDEX), pretrained=True):
        super(Resnet18_3d, self).__init__()
        # Load the pre-trained 3D CNN model (replace with your desired pre-trained model)
        # Accepts batched (B, T, C, H, W) trained on Kinetics 400
        self.model = models.video.r3d_18(pretrained=True)
                    
        # Freeze all layers
        for params in self.model.parameters():
            params.requires_grad = False
        
        if(args.input_learnable == True):
            self.model.stem[0].weight.requires_grad = True

        hidden_units1 = 512
        hidden_units2 = 512
        dropout_rate = args.dropout #default 0.3
        attn_dim = args.attn_dim #default 40
        out_features = 400 #400 for kinetics400
        
        #self.model.avgpool = nn.Identity() #nn.AdaptiveAvgPool3d((1, 6, 6))
        if(args.skip_attention == False):
            self.model.fc = nn.Sequential(
                nn.Flatten(),
                nn.TransformerEncoderLayer(d_model = 512, nhead = attn_dim, dim_feedforward = 512, dropout = dropout_rate, activation = 'relu'),
                nn.TransformerEncoderLayer(d_model = 512, nhead = attn_dim, dim_feedforward = 512, dropout = dropout_rate, activation = 'relu'),
            )
        else:
            self.model.fc = nn.Sequential(
                self.model.fc,
                nn.Linear(out_features, len(CATEGORY_INDEX),bias=True),
            )
        self.fc =nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.model(x) 
        x = self.fc(x)
        return x
    
def Resnet18_3D_With_Attention(args):
    return Resnet18_3d(args), AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
