from transformers import AutoImageProcessor, TimesformerForVideoClassification
from torch import nn
from Constants import CATEGORY_INDEX

def timeSformer600():

    processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k600")
    model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k600")

    for params in model.parameters():
        params.requires_grad = False

    model.classifier = nn.Linear(768,len(CATEGORY_INDEX),bias=True)

    return model,processor
