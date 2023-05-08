import math
import torch
from torch import nn, Tensor
import torchvision.models as models
from torchinfo import summary

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class EncodingBlock(nn.Module):
    
    def __init__(self,embed_dim):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x: Tensor) -> Tensor:
        
        

        return x

def Resnet18_2D_With_Attention():

    model = models.resnet18(weights='IMAGENET1K_V1')

    for params in model.parameters():
        params.requires_grad = False

    model.avgpool = nn.Flatten()

    model.fc = nn.Sequential(
        EncodingBlock()
        # PositionalEncoding(d_model=25088)
    )
    
    summary(model,input_size=(8,3,224,224))

    pass

    
Resnet18_2D_With_Attention()