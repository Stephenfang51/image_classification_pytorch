import torch.nn as nn
from utils import l2_norm



class Resnet18(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(Resnet18, self).__init__()
        self.backbone = model
        self.conv_last = nn.Conv2d(512, num_classes, 1)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = self.conv_last(x)
        x = x.view(x.size(0), -1)
        x = l2_norm(x)

        return (x)



class Se_resnext50_32x4d(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(Se_resnext50_32x4d, self).__init__()
        self.backbone = model
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.backbone.layer0(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        #         print(x.size())
        x = self.backbone.avg_pool(x)
        #         print(x.size())
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        #         x = self.backbone.last_linear(x)

        x = l2_norm(x)
        return x