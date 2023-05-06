from transformers import AutoImageProcessor, TimesformerForVideoClassification
from torch import nn

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18

from Constants import CATEGORY_INDEX

def timeSformer400():

    processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
    model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")

    for params in model.parameters():
        params.requires_grad = False

    model.classifier = nn.Linear(768,len(CATEGORY_INDEX),bias=True)

    return model,processor
    
def Resnet18WithAttention(args):

    # Define the custom image processor
    # def custom_image_processor(image):
    #     transforms = Compose([
    #         Resize(224, interpolation=2),  # Resize the shortest edge to 224
    #         CenterCrop(224),  # Perform center crop with the specified size
    #         ToTensor(),
    #         Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
    #         Lambda(lambda x: x * 0.00392156862745098)  # Apply the rescale factor
    #     ])
    #     return transforms(image)

    # Load the pre-trained 3D CNN model (replace with your desired pre-trained model)
    # Accepts batched (B, T, C, H, W) trained on Kinetics 400
    model = models.video.r3d_18(pretrained=True)
    #model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)

    #model.stem[0] = nn.Conv3d(8, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)

    # Freeze all layers
    for params in model.parameters():
        params.requires_grad = False

    hidden_units1 = 512
    hidden_units2 = 512
    dropout_rate = args.dropout #default 0.3
    attn_dim = args.attn_dim #default 40
    out_features = 400 #400 for kinetics400
    
    if(args.skip_attention == False):
        model.fc = nn.Sequential(
            nn.Flatten(),
            nn.TransformerEncoderLayer(d_model = 512, nhead = attn_dim, dim_feedforward = 512, dropout = dropout_rate, activation = 'relu'),
            nn.TransformerEncoderLayer(d_model = 512, nhead = attn_dim, dim_feedforward = 512, dropout = dropout_rate, activation = 'relu'),
            nn.Linear(512, len(CATEGORY_INDEX),bias=True),
            #nn.Linear(out_features, hidden_units1),
            #nn.ReLU(),
            #nn.Dropout(dropout_rate),
            #nn.Linear(hidden_units1, hidden_units2),
            #nn.ReLU(),
            #nn.Dropout(dropout_rate),
            #nn.Linear(hidden_units2, out_features)
        )
    else:
        model.fc = nn.Sequential(
            model.fc,
            nn.Linear(out_features, len(CATEGORY_INDEX),bias=True),
        )
    
    # Modify the last layer to match the number of output classes
    #model.classifier = nn.Linear(out_features, len(CATEGORY_INDEX))
    #return model, custom_image_processor
    return model, AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")

def getModel(args):
    print("Proceeding with Model:", args.model)
    if(args.model == "timesformer400"):
        return timeSformer400()
    elif(args.model == "resnet18WithAttention"):
        return Resnet18WithAttention(args)