from transformers import AutoImageProcessor, TimesformerForVideoClassification
from torch import nn

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18

from Constants import CATEGORY_INDEX

import math
import numpy as np

from einops import rearrange, reduce, repeat

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batchSize: int = 8):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.cls_token = nn.Parameter(torch.zeros(batchSize, 1, d_model))

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = torch.cat((self.cls_token, x), dim=1)
        
        x = x + self.pe[:x.size(0)]
        # return self.dropout(x)
        return x

class Resnet18_3d(nn.Module):
    def __init__(self, args, num_classes=len(CATEGORY_INDEX), pretrained=True):
        super(Resnet18_3d, self).__init__()
        # Load the pre-trained 3D CNN model (replace with your desired pre-trained model)
        # Accepts batched (B, C, T, H, W) trained on Kinetics 400
        if(args.model == "resnet18WithAttention"):
            self.model = models.video.r3d_18(pretrained=True)
        elif(args.model == "resnet182Plus1"):
            self.model = models.video.r2plus1d_18(pretrained=True)

        self.skip_attention = args.skip_attention
        
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
            #self.model.avgpool = nn.AdaptiveAvgPool3d((1, 2, 2))
            self.model.fc = nn.Sequential(
                nn.Flatten(),
                nn.TransformerEncoderLayer(d_model = 512, nhead = attn_dim, dim_feedforward = 512, dropout = dropout_rate, activation = 'gelu'),
                nn.TransformerEncoderLayer(d_model = 512, nhead = attn_dim, dim_feedforward = 512, dropout = dropout_rate, activation = 'gelu'),
            )
            self.fc =nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, num_classes),
        )
        else:
            self.model.fc = nn.Sequential(
                self.model.fc,
                nn.Linear(out_features, len(CATEGORY_INDEX),bias=True),
            )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.model(x) 
        if(self.skip_attention == False):
            x = self.fc(x)
        return x
    
def Resnet18_3D_With_Attention(args):
    return Resnet18_3d(args), AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
