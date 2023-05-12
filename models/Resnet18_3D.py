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

        self.d_model = 512
        self.ntoken = 100
        self.model = models.video.r3d_18(pretrained=True)
        self.pos_encoder = PositionalEncoding(512, 0.1)
        self.encoder = nn.Embedding(8, self.d_model)
        self.decoder = nn.Linear(self.d_model, 8)
        
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
        self.model.fc = nn.Identity()
        if(args.skip_attention == False):
            self.transformer = nn.Sequential(
                #nn.Flatten(),
                nn.TransformerEncoderLayer(d_model = 512, nhead = attn_dim, dim_feedforward = 512, dropout = dropout_rate, activation = 'relu'),
                nn.TransformerEncoderLayer(d_model = 512, nhead = attn_dim, dim_feedforward = 512, dropout = dropout_rate, activation = 'relu'),
            )
        else:
            self.transformer = nn.Sequential(
                self.model.fc,
                nn.Linear(out_features, len(CATEGORY_INDEX),bias=True),
            )
        self.fc =nn.Sequential(
                nn.Flatten(),
                nn.Linear(4104, num_classes,bias=True),
        )
    
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, x):        
        x = self.model(x) 
        x = torch.tensor(x).to(torch.int64)
        x = self.encoder(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x) # check src mask
        x = self.decoder(x)
        x = self.fc(x)
        return x
    
def Resnet18_3D_With_Attention(args):
    return Resnet18_3d(args), AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
