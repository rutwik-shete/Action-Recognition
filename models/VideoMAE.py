import torch
from torch import nn
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, RandomRotation, Resize
from torchvision.transforms.functional import crop
from transformers import AutoImageProcessor, TimesformerForVideoClassification
from Constants import CATEGORY_INDEX
import torch.nn.init as init


def VideoMAE(dropout_rate=0.5):
    processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
    model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")

    num_features_before_fcnn = model.classifier.in_features  # obtain the number of input features to the classifier

    # Replace the old classifier with the new one that includes dropout and batch normalization.
    model.classifier = nn.Sequential(
        nn.Linear(num_features_before_fcnn, len(CATEGORY_INDEX), bias=True),  # Final classifier
        nn.LayerNorm(len(CATEGORY_INDEX))  # Layer normalization
    )

    # Add a 13D Convolutional layer
    model.features = nn.Sequential(
        nn.Conv3d(3, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),  # 13D Convolution
        nn.BatchNorm3d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # Temporal pooling
    )

    # Xavier initialization of weights
    def weights_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

    model.apply(weights_init)

    # Apply mask to the model
    mask_percentage = 0.9  # Set the desired mask percentage
    mask = torch.rand(model.classifier[0].weight.shape) < mask_percentage
    model.classifier[0].weight.data *= mask.float()

    # Freeze the parameters of the pretrained model
    for param in model.parameters():
        param.requires_grad = False

    # Enable gradient calculation for the new classifier
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Data augmentation transformations
    data_transforms = Compose([
        Resize((224, 224)),  # Resize the frame to a fixed size of 224
        RandomCrop((192, 192)),  # Randomly crop the frame to 192x192
        RandomHorizontalFlip(),  # Randomly flip the frame horizontally
        RandomRotation(15),  # Randomly rotate the frame by up to 15 degrees
        nn.Sequential(
            nn.Upsample((10, 224, 224)),  # Upsample the frame to 10 frames
            nn.Softmax(dim=1)
        )  # Apply softmax to obtain a probability distribution
    ])

    # Apply data augmentation transformations to the processor
    processor.transform = data_transforms

    return model, processor




