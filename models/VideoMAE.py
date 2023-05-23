from transformers import AutoImageProcessor, TimesformerForVideoClassification
from torch import nn
from Constants import CATEGORY_INDEX
import torch
import types


def VideoMAE():
    processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
    model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")

    for params in model.parameters():
        params.requires_grad = False

    model.classifier = nn.Linear(768, len(CATEGORY_INDEX), bias=True)

    # Define the mask tensor
    total_frames = 8  # Total number of frames in the input
    num_masked_frames = int(total_frames * 0.85)  # 80% of the frames to mask
    mask_tensor = torch.cat(
        [torch.zeros((1, num_masked_frames), dtype=torch.float32), torch.ones((1, total_frames - num_masked_frames), dtype=torch.float32)],
        dim=1
    )

    # Apply the mask to the input frames
    def forward(self, frames):
        masked_frames = frames * mask_tensor
        processed_frames = processor(masked_frames)
        return processed_frames

    # Monkey-patch the processor's forward function with the modified version
    processor.forward = types.MethodType(forward, processor)

    return model, processor