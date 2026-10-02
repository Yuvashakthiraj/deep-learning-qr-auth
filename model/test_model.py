import torch
from mobilenet_model import get_mobilenet_model

model = get_mobilenet_model()

dummy_input = torch.randn(1, 3, 224, 224)
output = model(dummy_input)

print("Output shape:", output.shape)
