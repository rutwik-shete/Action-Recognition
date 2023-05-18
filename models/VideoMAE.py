from transformers import AutoImageProcessor, TimesformerForVideoClassification
from torch import nn
from Constants import CATEGORY_INDEX


class VideoMAE(nn.Module):
    def __init__(self):
        super(VideoMAE, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
        self.backbone, self.processor = timeSFormer400()
        self.encoder = nn.Identity()  # Example encoder layer
        self.decoder = nn.Identity()  # Example decoder layer
        # self.classifier = nn.Linear(768, len(CATEGORY_INDEX), bias=True)

    def forward(self, x):
        # processed_x = self.processor(x)
        processed_x = x
        video_features = self.backbone(processed_x)
        video_features = video_features.logits
        encoded_features = self.encoder(video_features)
        reconstructed_video = self.decoder(encoded_features)
        output = reconstructed_video

        return output

def timeSFormer400():
    processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
    model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Linear(768, len(CATEGORY_INDEX), bias=True)

    return model, processor

def video_mae_model(args):
    return VideoMAE(), None