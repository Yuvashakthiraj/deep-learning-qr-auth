import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


def get_mobilenet_model():
    """
    Load a MobileNetV2 model pre-trained on ImageNet and adapt it
    for binary QR code classification (Original vs Fake).

    The feature extractor layers are frozen to preserve the
    ImageNet-learned representations; only the final classifier
    layer is trained from scratch.
    """
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights)

    # Freeze feature extractor — leverage pre-trained ImageNet features
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the ImageNet classifier (1280 → 1000) with our binary head
    model.classifier[1] = nn.Linear(
        in_features=1280,
        out_features=2  # 0 = Original, 1 = Fake
    )

    return model
