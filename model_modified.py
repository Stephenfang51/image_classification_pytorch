
from model.model_zoo import get_model
import torch
from torch import nn
from torchsummary import summary
from model.repvgg import Net
from utils.margin import ArcMarginProduct


margin = ArcMarginProduct(2048, 6, 32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0

model = get_model('RepVGG-B1g2')()
model = Net(model)

x = torch.randn((256, 3, 224, 224), dtype=torch.float32)
label = torch.randn((256, 3, 224, 224))
x = model(x)
output = margin(x, label)
print('outputsize', output.shape)


print('size:', (model(x)).shape)
