from transformers import AutoImageProcessor, TimesformerForVideoClassification
from torch import nn

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchinfo as summary

def Resnet18_2D_With_Attention():

    model = models.resnet18(models.ResNet18_Weights)

    for params in model.parameters():
        params.requires_grad = False

    model.avgpool = nn.Flatten()

    model.fc = nn.Sequential(
        nn.MultiheadAttention(embed_dim=25088,num_heads=8)
    )


    # image_transforms = models.ResNet18_Weights.IMAGENET1K_V1.transforms
    summary(model,input_size=(8,3,224,224))
    
    # return model
    pass

Resnet18_2D_With_Attention() 