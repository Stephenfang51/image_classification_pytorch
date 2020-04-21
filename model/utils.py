from model import efficientNet, resnet
from train_detail import train_detail
import pretrainedmodels
from torchvision import models
from model.efficientNet import EfficientNet
from model.resnet import Resnet18, Resnet50, Se_resnext50_32x4d
from model.resnest import *

# train_opt_2 = train_detail.parse()
# Network_name = train_opt_2.model

def select_network(network, num_classes = 1000):


    if network == 'resnet18':
        backbone = models.resnet18(pretrained=True)
        model = Resnet18(backbone, num_classes=num_classes)
        return model
    #         metric_fc = ArcMarginProduct(512, Num_classes, s=30, m=0.5, easy_margin=False)
    elif network == 'resnet50':
        backbone = models.resnet50(pretrained=True)
        model = Resnet50(backbone, num_classes=num_classes)
        return model

    elif network == 'se_resnext50_32x4d':
        backbone = pretrainedmodels.se_resnext50_32x4d(pretrained='imagenet')
        model = Se_resnext50_32x4d(backbone, num_classes=num_classes)
        return model

    elif network == 'resnest50':
        model = resnest50(pretrained = True, num_classes=num_classes)
        # model = resnest50(num_classes=num_classes)
        return model

    elif network == 'resnest101':
        model = resnest101(pretrained = True, num_classes=num_classes)
        # model = resnest101(num_classes=num_classes)
        return model


    elif network == 'efficientb0':
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
        return model

    elif network == 'efficientb5':
        model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=num_classes)
        return model


    # return model



def freeze_param(network, model):
    if network == 'resnet18':
        for p in model.backbone.layer0.parameters(): p.requires_grad = False
        for p in model.backbone.layer1.parameters(): p.requires_grad = False
        for p in model.backbone.layer2.parameters(): p.requires_grad = False
        for p in model.backbone.layer3.parameters(): p.requires_grad = False

        return model

    elif network == 'resnet50':
        for p in model.backbone.layer0.parameters(): p.requires_grad = False
        for p in model.backbone.layer1.parameters(): p.requires_grad = False
        for p in model.backbone.layer2.parameters(): p.requires_grad = False
        for p in model.backbone.layer3.parameters(): p.requires_grad = False

        return model

    elif network == 'se_resnext50_32x4d':

        for p in model.backbone.layer0.parameters(): p.requires_grad = False
        for p in model.backbone.layer1.parameters(): p.requires_grad = False
        for p in model.backbone.layer2.parameters(): p.requires_grad = False
        for p in model.backbone.layer3.parameters(): p.requires_grad = False

        return model

    elif network == 'resnest50':
        for p in model.layer1.parameters(): p.requires_grad = False
        for p in model.layer2.parameters(): p.requires_grad = False
        for p in model.layer3.parameters(): p.requires_grad = False
        for p in model.layer4.parameters(): p.requires_grad = False

        return model

    elif network == 'resnest101':
        for p in model.layer1.parameters(): p.requires_grad = False
        for p in model.layer2.parameters(): p.requires_grad = False
        for p in model.layer3.parameters(): p.requires_grad = False
        for p in model.layer4.parameters(): p.requires_grad = False

        return model



    elif network == 'efficientb0':
        for p in model.parameters():
            p.requires_grad = False
        model._fc.weight.requires_grad = True
        model._fc.bias.requires_grad = True

        return model


    elif network == 'efficientb5':
        for p in model.parameters():
            p.requires_grad = False
        model._fc.weight.requires_grad = True
        model._fc.bias.requires_grad = True

        return model
