import torch
import torch.nn as nn
import torchvision.models as models
from Constants import CATEGORY_INDEX
from transformers import AutoImageProcessor


class DINO(nn.Module):
    def __init__(self, args, num_classes=len(CATEGORY_INDEX)):
        super(DINO, self).__init__()
        self.teacher = Resnet18_3d(args, num_classes, pretrained=True)
        self.student = Resnet18_3d(args, num_classes, pretrained=True)
        self.classifier = nn.Linear(512, num_classes)

        self.loging_freq = args.loging_freq
        self.momentum_teacher = args.momentum_teacher
        self.n_crops = args.n_crops
        self.out_dim = args.out_dim
        self.clip_grad = args.clip_grad
        self.norm_last_layer = args.norm_last_layer
        self.teacher_temp = args.teacher_temp
        self.student_temp = args.student_temp

    def forward(self, x):
        num_dims = len(x.shape)

        if num_dims == 4:
            # If the input tensor has shape (B, C, H, W)
            B, C, H, W = x.shape
            T = 1
            x = x.permute(0, 2, 1, 3, 4)  # Permute the dimensions to (B, T=1, C, H, W)
        elif num_dims == 5:
            # If the input tensor has shape (B, T, C, H, W)
            B, T, C, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4)  # Permute the dimensions to (B, C, T, H, W)
        else:
            raise ValueError("Invalid input shape. Expected 4 or 5 dimensions.")

        print("Input shape:", x.shape)

        x = x.view(B * T, C, H, W)  # Reshape to (B*T, C, H, W)

        print("Reshaped input shape:", x.shape)

        teacher_output = self.teacher(x)  # Pass the input through the teacher network
        student_output = self.student(x)  # Pass the input through the student network

        teacher_output = teacher_output.view(B, T, -1)  # Reshape to (B, T, -1)
        student_output = student_output.view(B, T, -1)  # Reshape to (B, T, -1)

        teacher_output = teacher_output.permute(0, 2, 1)  # Permute the dimensions to (B, -1, T)
        student_output = student_output.permute(0, 2, 1)  # Permute the dimensions to (B, -1, T)

        teacher_output = teacher_output.mean(dim=2)  # Take the mean across the T dimension
        student_output = student_output.mean(dim=2)  # Take the mean across the T dimension

        print("Teacher output shape:", teacher_output.shape)
        print("Student output shape:", student_output.shape)

        teacher_logits = self.classifier(teacher_output)  # Pass the teacher output through the classifier
        student_logits = self.classifier(student_output)  # Pass the student output through the classifier

        print("Teacher logits shape:", teacher_logits.shape)
        print("Student logits shape:", student_logits.shape)

        return teacher_logits, student_logits




def Resnet18_3d(args, num_classes=len(CATEGORY_INDEX), pretrained=True):
    model = models.video.r3d_18(pretrained=pretrained)

    # Modify the stem convolutional layer to have 32 input channels
    model.stem[0] = nn.Conv3d(32, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)

    # Modify the last convolutional layer to have 3 output channels
    model.layer4[1].conv2 = nn.Conv3d(256, 3, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)

    # Freeze all layers
    for params in model.parameters():
        params.requires_grad = False

    hidden_units1 = 512
    hidden_units2 = 512
    dropout_rate = args.dropout
    attn_dim = args.attn_dim
    out_features = 400

    if not args.skip_attention:
        model.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(512, hidden_units1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_units1, hidden_units2),
            nn.ReLU(inplace=True),
        )
    else:
        model.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes, bias=True),
        )

    return model

def DINOModel(args):
    model = DINO(args)
    processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
    return model, processor

