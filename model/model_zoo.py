from resnest.torch import *
from torchvision.models import *
import timm
from .repvgg import *


model_zoo = {
    'resnet18':resnet18,
    'resnet50':resnet50,
    'resnet101':resnet101,

    'resnest50':resnest50,
    'resnest101': resnest101,
    'resnest50_fast_1s1x64d' : resnest50_fast_1s1x64d,
    'resnest50_fast_2s1x64d' : resnest50_fast_2s1x64d,
    'resnest50_fast_4s1x64d' : resnest50_fast_4s1x64d,
    'resnest50_fast_1s2x40d':resnest50_fast_1s2x40d,
    'resnest50_fast_2s2x40d':resnest50_fast_2s2x40d,
    'resnest50_fast_4s2x40d':resnest50_fast_4s2x40d,
    'resnest50_fast_1s4x24d':resnest50_fast_1s4x24d,

    'RepVGG-A0': create_RepVGG_A0,
    'RepVGG-A1': create_RepVGG_A1,
    'RepVGG-A2': create_RepVGG_A2,
    'RepVGG-B0': create_RepVGG_B0,
    'RepVGG-B1': create_RepVGG_B1,
    'RepVGG-B1g2': create_RepVGG_B1g2,
    'RepVGG-B1g4': create_RepVGG_B1g4,
    'RepVGG-B2': create_RepVGG_B2,
    'RepVGG-B2g2': create_RepVGG_B2g2,
    'RepVGG-B2g4': create_RepVGG_B2g4,
    'RepVGG-B3': create_RepVGG_B3,
    'RepVGG-B3g2': create_RepVGG_B3g2,
    'RepVGG-B3g4': create_RepVGG_B3g4,

    'tf_efficientnet_b4_ns':timm.create_model('tf_efficientnet_b4_ns', True)
    
}



def get_model(model_name):
    """
    :param model_name: from cfg
    :type model_name: str
    :return: model
    :rtype:
    """
    if model_name in model_zoo:
        return model_zoo[model_name]


