import math
import torch
from torch import nn, Tensor
import torchvision.models as models
from torchinfo import summary

from transformers import AutoImageProcessor, TimesformerForVideoClassification

from Constants import CATEGORY_INDEX
# from vit_utils import to_2tuple
from einops import rearrange, reduce, repeat

# class PatchEmbed(nn.Module):
#     """ Image to Patch Embedding
#     """
#     def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = num_patches

#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

#     def forward(self, x):
#         B, C, T, H, W = x.shape
#         x = rearrange(x, 'b c t h w -> (b t) c h w')
#         x = self.proj(x)
#         W = x.size(-1)
#         x = x.flatten(2).transpose(1, 2)
#         return x, T, W

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batchSize: int = 24):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.cls_token = nn.Parameter(torch.zeros(1 ,1, d_model))

        # self.pos_embed = None

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

        my_cls_token = self.cls_token.expand((x.size(0), -1, -1))

        x = torch.cat((my_cls_token, x), dim=1)

        x = x + self.pe[:x.size(0)]
        # return self.dropout(x)
        return x
    
class BatchFlattening(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        
        x = rearrange(x, 'b w -> (b w)')
        return x

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.resnet50(weights='IMAGENET1K_V1')
        self.batchSize = 8

        for params in self.model.parameters():
            params.requires_grad = False

        self.model.conv1.weight.requires_grad = True

        self.model.avgpool =  nn.AdaptiveAvgPool2d((1, 1))

        self.model.fc = nn.Flatten()

        self.midLayer = nn.Sequential(
            PositionalEncoding(d_model=2048,batchSize=self.batchSize),
            nn.TransformerEncoderLayer(d_model=2048,nhead=16,dim_feedforward = 2048,activation = 'gelu',dropout=0.3),
            nn.Flatten(),
            # BatchFlattening(),
            # 18432 34816
            nn.Linear(18432,len(CATEGORY_INDEX),bias=True)
        )

    def forward(self, x: Tensor) -> Tensor:

        B,T,C,H,W = x.shape

        self.batchSize = B

        x = rearrange(x, 'b t c h w -> (b t) c h w')
        
        x = self.model(x)

        x = rearrange(x, '(b t) d -> b t d',b=B,t=T)

        x = self.midLayer(x)

        return x

def Resnet50_2D_With_Attention():

    
    model = CustomModel()

    # summary(model.to(torch.device("mps")), input_size=(8 , 8, 3, 224, 224) , device=torch.device("mps"))

    # summary(model, input_size=(2 , 8, 3, 224, 224) )

    return model, AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")

    
# Resnet18_2D_With_Attention()