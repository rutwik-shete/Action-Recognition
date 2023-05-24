from transformers import AutoImageProcessor, TimesformerForVideoClassification
from torch import nn

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18

from Constants import CATEGORY_INDEX

from models.Resnet18_2D import Resnet18_2D_With_Attention
from models.Resnet18_3D import Resnet18_3D_With_Attention
from models.Timesformer600 import timeSformer600
from models.Resnet18_2D import Resnet18_2D_With_Attention
from models.VideoMAE import VideoMAE
import torch.nn.init as init


def timeSformer400():
    processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
    model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")

    for params in model.parameters():
        params.requires_grad = True

    model.classifier = nn.Linear(768,len(CATEGORY_INDEX),bias=True)


def getModel(args):
    print("Proceeding with Model:", args.model)
    if(args.model == "timesformer400"):
        return timeSformer400()
    elif(args.model == "timesformer600"):
        return timeSformer600()
    elif(args.model == "2Dresnet18"):
        return Resnet18_2D_With_Attention()
    elif(args.model == "resnet18WithAttention" or args.model == "resnet182Plus1"):
        return Resnet18_3D_With_Attention(args)
    elif(args.model == "videomae"):
        return VideoMAE()
    
  


