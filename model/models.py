import yaml
from torch import nn
import timm
import torch
import os
# from config import CFG

#loading config
# with open(args.config, erros='ignore') as file:
#     CFG = yaml.safe_load(file) #now y is a dict-like object



model_path = '/home/liwei.fang/.cache/torch/hub/checkpoints/'

#EfficientNet
class Classifier_efficientnet(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x


#for ResNet and ResNest
class Classifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = model_arch(pretrained) #instance model
        #changed the classifier layer to be num of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, n_class)
        # num_features = self.model.fc.out_features #why not out_features
        # self.model.classifier = nn.Linear(num_features, n_class) #add classifier


    def forward(self, x):
        x = self.model(x)
        return x

#for Repvgg
class Classifier_repvgg(nn.Module):
    def __init__(self, cfg='./config.yaml', model_arch='RepVGG-B1g2', n_class=80, pretrained=False, model_name='RepVGG-B1g2'):
        super().__init__()
        
        with open(cfg, erros='ignore') as file:
            CFG = yaml.safe_load(file) #now y is a dict-like object
        self.model = model_arch() #instance model
        #changed the classifier layer to be num of classes
        if pretrained:
            state_dict = torch.load(model_path + model_name + '-train'+ '.pth')
            self.model.load_state_dict(state_dict)
        num_features = self.model.linear.in_features
        self.model.linear = nn.Linear(num_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x


def modified_classifier_output(cfg, model_fn, args):
    
    class_num = len(os.listdir(args.train_img_path))

    if cfg['model_arch'][0:6] == 'resnet' or cfg['model_arch'][0:7] == 'resnest':
        return Classifier(model_fn, class_num, pretrained=cfg['pretrained'])
    elif cfg['model_arch'][0:3] == 'Rep': #repvgg
        return Classifier_repvgg(model_fn, class_num, cfg['pretrained'])
    elif cfg['model_arch'][0:2] == 'tf':
        return Classifier_efficientnet(model_fn, class_num, cfg['pretrained'])



