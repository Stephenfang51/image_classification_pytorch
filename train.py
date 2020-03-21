# import argparse
import os
import random
import time
import matplotlib.pyplot as plt
from dataset import CDCDataset
import torch as t
from torchvision import transforms
from loss import Focalloss, LabelSmoothSoftmaxCE
from torchvision import models
from torch.utils.data import DataLoader
from train_detail import train_detail

import torch.nn as nn
# from model.resnet import Resnet18, Se_resnext50_32x4d
# from model.efficientNet import EfficientNet
from utils import accuracy, GradualWarmupScheduler, select_sehcduler, opcounter
from model.utils import select_network, freeze_param


train_opt = train_detail().parse()

#train setting
train_path = train_opt.train_path
Num_classes = train_opt.num_classes
Size = train_opt.input_size
Model = train_opt.model
Checkpoint = train_opt.checkpoints
Resume = train_opt.resume
Loss = train_opt.loss
Num_epochs = train_opt.num_epochs
Batch_size = train_opt.batch_size
Freeze = train_opt.freeze
Init_lr = train_opt.init_lr

# Lr scheduler setting
Lr_scheduler = train_opt.lr_scheduler
Step_size = train_opt.step_size

#Grad_Warm setting
Multiplier = train_opt.multiplier
Total_epoch = train_opt.total_epoch
# Grad_warm = train_opt.warm_up


#focal loss
Alpha = train_opt.alpha
Gamma = train_opt.gamma


#data aug
Re = train_opt.re







def train():
    # -------计算时间----------#
    begin_time = time.time()

    # ------确认Model---------#
    model = select_network(Model, Num_classes)

    opcounter(model)
    model.cuda()
    #     metric_fc.cuda()

    # freeze layers
    if Freeze:
        model = freeze_param(Model, model)

    # ------train dataset trasforms----#
    train_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.ColorJitter(0.05, 0.05, 0.05),
                                          transforms.RandomRotation(30),
                                          transforms.Resize((Size, Size)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
    # ------ train CDC ------#
    dst_train = CDCDataset(ims2labels, transform=train_transform, re = Re)
    # ------ pytorch Dataloader for trai------#
    dataloader_train = DataLoader(dst_train, batch_size=Batch_size // 2, shuffle=True, num_workers=8)

    # -------load checkpoints -------------#
    if Resume:
        model = t.load(os.path.join('./checkpoints/{}.pth'.format(Checkpoint)))

    # train
    sum = 0
    train_loss_sum = 0
    train_top1_sum = 0

    # ------- set loss fuction -----------#
    if Loss == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss().cuda()
    elif Loss == 'FocalLoss':
        criterion = Focalloss(Num_classes, alpha=Alpha, gamma=Gamma, size_average=True)
    elif Loss == 'LabelSmoothSoftmaxCE':
        criterion = LabelSmoothSoftmaxCE(lb_pos=0.9, lb_neg=0.05)

    #     criterion.cuda()
    # ---------optimizer -----------------#
    optimizer = t.optim.Adam([{'params': filter(lambda p: p.requires_grad, model.parameters())},
                              #                                   {'params':metric_fc.parameters()}
                              ], lr=Init_lr, betas=(0.9, 0.999), weight_decay=0.002)

    # --------scheduler ------------------#

    sehcduler = select_sehcduler(Lr_scheduler, optimizer, Step_size, Multiplier, Total_epoch, Num_epochs)

    print('Start training')
    # ------ list for plot curve ----------#
    train_loss_list, val_loss_list = [], []
    train_top1_list, val_top1_list = [], []

    lr_list = []
    # -------- train --------------------#
    for epoch in range(Num_epochs):
        sehcduler.step()

        ep_start = time.time()
        model.train()
        top1_sum = 0

        for im, (ims, labels) in enumerate(dataloader_train):
            input = ims.cuda()
            target = labels.cuda().long()

            #             feature = model(input)
            #             output = metric_fc(feature, target)
            output = model(input)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            top1 = accuracy(output.data, target.data, topk=(1,))
            train_loss_sum += loss.data.cpu().numpy()
            train_top1_sum += top1[0]
            sum += 1
            top1_sum += top1[0]

        lr = optimizer.state_dict()['param_groups'][0]['lr']  # 取得当前的lr
        print('Epoch [%d / %d ] | lr=[%f] | training loss [%.4f] | train top1 [%.4f] | time[%.4f s]' \
              % (epoch + 1, Num_epochs, lr, train_loss_sum / sum, train_top1_sum / sum, time.time() - ep_start))

        train_loss_list.append(train_loss_sum / sum)
        train_top1_list.append(train_top1_sum / sum)
        lr_list.append(lr)

        sum = 0
        train_loss_sum = 0
        train_top1_sum = 0
        # -----------以上归零-------------------#
        if (epoch + 1) % 50 == 0 and epoch < Num_epochs or (epoch + 1) == Num_epochs:
            t.save(model, './checkpoints/{}.pth'.format(Checkpoint))
            print('saving model successfully')

        # -------------计算钟头-----------------#
        if (time.time() - begin_time) / 60 / 60 > 8:  # 超过8小 中断
            break
    # --------------inference time ------------#
    figs = plt.figure()
    fig1 = figs.add_subplot(3, 1, 1)
    fig2 = figs.add_subplot(3, 1, 2)
    fig3 = figs.add_subplot(3, 1, 3)
    x = [i for i in range(len(train_loss_list))]  # 求出x label的值

    fig1.plot(x, train_loss_list, label='train loss')
    fig1.legend(loc='upper right')
    fig2.plot(x, train_top1_list, label='train_top1 accuracy')
    fig2.legend(loc='bottom right')
    fig3.plot(x, lr_list, label='lr')
    fig3.legend(loc='upper right')

    plt.show()
    plt.savefig('training_result')

        # print('start inference')
        # inference(model)


if __name__ == '__main__':


    # args = parser.parse_args()

    if train_path :
        # train_path = args.train_path
        cls2label = {}
        for label, cls in enumerate(os.listdir(train_path)):
            cls2label[cls] = label
        print(cls2label)
        # %%
        ims2labels = {}
        ims2labels_train = {}
        ims2labels_val = {}
        for cls in os.listdir(train_path):
            im_num = len(os.listdir(os.path.join(train_path, cls)))
            # total ims
            for im in os.listdir(os.path.join(train_path, cls)):
                impath = os.path.join(train_path, cls, im)  # 取得train下 每个cls下每个图档path
                ims2labels[impath] = cls2label[cls]
                # 将类别的标签赋值给图片路径，也就是ims2labels 包含所有train的图片路径及对应的label

            # 定义验证集, 从total train中取一成当做验证集
            val_ims = random.sample(os.listdir(os.path.join(train_path, cls)), int(im_num * 0.1))
            #     print(val_ims)
            for im in val_ims:  # 从随机取出的val中定义val每个图片路径+label
                impath = os.path.join(train_path, cls, im)
                ims2labels_val[impath] = cls2label[cls]
            for im in os.listdir(os.path.join(train_path, cls)):
                if im not in val_ims:  # 将剩余不是val的，定义为train
                    impath = os.path.join(train_path, cls, im)
                    ims2labels_train[impath] = cls2label[cls]
        # print('total:', list(ims2labels.items())[:5], len(list(ims2labels.items())))
        # print('train:', list(ims2labels_train.items())[:5], len(list(ims2labels_train.items())))
        # print('validation:', list(ims2labels_val.items())[:5], len(list(ims2labels_val.items())))
    else:
        print("please set up a train dataset path")

    train()



        # inference_TTA(model)






