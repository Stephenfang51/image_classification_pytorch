"""
code from https://github.com/Lyken17/pytorch-OpCounter
"""


from torchvision.models import resnet18
from model.model_zoo import get_model
import torch
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0

# model = resnet18()
# model = get_model('resnest50')()
model = get_model('RepVGG-B1g2')()

model.to(device)
# input = torch.randn(1, 3, 224, 224).to(device)
summary(model, input_size=(3, 224, 224))




# resnest50 = get_model('resnest50')
# model = resnest50()




# macs, params = profile(model, inputs=(input,), verbose=False)
# print(macs, params)

# macs, params = clever_format([macs, params], "%.3f")
# print(macs, params)