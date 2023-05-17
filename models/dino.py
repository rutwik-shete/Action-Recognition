import torch
import torch.nn as nn
from torch import Tensor
import torchvision.models as models
from torchvision.transforms import transforms
from Constants import CATEGORY_INDEX
from data_manager import BlockFrameDataset
import timm
from torchvision.transforms.functional import to_pil_image



def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # Generate samples from a normal distribution which follows N(mean, std^2)
    r = tensor.new_tensor(torch.randn(tensor.shape))
    # Apply clip to keep values within specified range (a, b)
    clipped = r.clamp(min=a, max=b)
    # Scale and shift the output
    return mean + std * clipped


class DINO(nn.Module):
    def __init__(self, args, num_classes=len(CATEGORY_INDEX)):
        super(DINO, self).__init__()
        self.args = args
        print("Initializing DINO model...")
        self.teacher = self.create_vit_model()  # Load the pre-trained ViT as the teacher
        self.student = nn.Sequential(*list(self.teacher.children())[:-1])  # Use the loaded ViT as the student, but without the classifier
        self.augmentations = self.get_augmentations()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.size()

        # Reshape the input tensor
        x = x.permute(0, 2, 1, 3, 4)  # Permute dimensions to (B, T, C, H, W)

        print("Processing input...")
        x1 = self.augmentations(x.clone())  # Apply some augmentations to x
        x2 = self.augmentations(x)  # Apply different augmentations to x

        print("Passing input through teacher network...")
        teacher_output = self.teacher(x1)  # Pass the first input through the teacher network
        print("Teacher output shape:", teacher_output.shape)

        print("Passing input through student network...")
        student_output = self.student(x2)  # Pass the second input through the student network
        print("Student output shape:", student_output.shape)

        # Reshape the outputs to match the input_size
        teacher_output = teacher_output.view(B, T, -1, 7, 7)
        student_output = student_output.view(B, T, -1, 7, 7)

        return teacher_output, student_output



    def create_vit_model(self):
        print("Creating Vision Transformer model...")
        model = timm.create_model('vit_base_patch16_224', pretrained=True)

        # Replace the last layer of the Vision Transformer model
        if hasattr(model, "head"):
            num_ftrs = model.head.in_features  # get the number of input features for the head layer
            model.head = nn.Linear(num_ftrs, len(CATEGORY_INDEX))  # replace the head layer
        else:
            raise AttributeError("The model does not have a suitable attribute to replace the last layer.")

        # Initialize the last layer with truncated normal distribution
        trunc_normal_(model.head.weight, std=0.02)
        nn.init.constant_(model.head.bias, 0)

        return model

    def get_augmentations(self):
        print("Creating data augmentations...")
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.1, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(3, sigma=(0.1, 2.0)),  # Gaussian blur as in DINO paper
        ]

        if self.args.model == "dino":
            augmentation.insert(0, transforms.ToPILImage())  # Convert tensor to PIL Image

        augmentation.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )  # Normalization values for ImageNet
        ])

        return transforms.Compose(augmentation)



def DINOModel(args):
    print("Creating DINO model and processor...")
    model = DINO(args)
    return model

