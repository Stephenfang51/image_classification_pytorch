import torch
import pandas as pd
import os
import numpy as np
import time
import argparse
from torch import nn
from tqdm import tqdm
from data_load.data import prepare_dataloader
from data_load.data import Classification_Dataset
from data_load.data_transforms import get_train_transforms, get_valid_transforms
from model.models import Classifier, Classifier_efficientnet, Classifier_repvgg
from model.model_zoo import get_model
from tensorboardX import SummaryWriter
from utils.fusion_matrix import calc_cmtx, save_cmtx
from utils.loss import CrossEntropyLabelSmooth
from utils.margin import ArcMarginProduct


#
from model.repvgg import Net


"""
This version did not divide training set into 8:2 validation set
need to provide a new validation set

python train_.py --train_img_path data/spc_smokingcalling_20210120_2/ --train_csv data/spc_smokingcalling_20210120_2.csv --valid_img_path data_inference/spc_head_test_20210120/ --valid_csv data_inference/spc_head_test_20210120.csv
python train_arcface.py --train_img_path data/spc_smokingcalling_20210207_small/ --train_csv data/spc_smokingcalling_20210207_small.csv --valid_img_path data_inference/spc_head_test_20210203/ --valid_csv data_inference/spc_head_test_20210203.csv


"""
"""---options---"""
parser = argparse.ArgumentParser(description='model training')
parser.add_argument('--train_img_path', type=str, required=True, help='training data path')
parser.add_argument('--valid_img_path', type=str, required=True, help='valid data path')
parser.add_argument('--train_csv', type=str, required=True, help='training csv file to load')
parser.add_argument('--valid_csv', type=str, required=True, help='valid csv file to load')
parser.add_argument('--cal_mtx', type=bool, default=True, help="whether to calculate fusion matrix")


args = parser.parse_args()



def train_every_epoch(epoch, model, margin, loss_fn, optimizer, train_dataloader, device, scheduler=None, schd_batch_update=False, running_loss = None):
    model.train()
    pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        # print('nan or not', np.any(np.isinf(imgs.cpu().numpy())))
        images_labels = image_labels.to(device).long()
        raw_logits = model(imgs)

        output = margin(raw_logits, image_labels)
        loss_thisepoch = loss_fn(raw_logits, images_labels)
        loss_thisepoch.backward()

        if running_loss == None:
            running_loss = loss_thisepoch.item()
        else :
            running_loss = running_loss * .99 + loss_thisepoch * .01
        if ((step + 1)%CFG['accum_iter'] == 0) or ((step+1) == len(train_dataloader)):
            optimizer.step()
            optimizer.zero_grad()

            if scheduler != None and schd_batch_update:
                scheduler.step()


        if ((step + 1) % CFG['verbose_step'] == 0) or((step + 1) == len(train_dataloader)):
            description = f'epoch {epoch} loss: {running_loss : .4f}'
            pbar.set_description(description)

    if scheduler != None and not schd_batch_update:
        scheduler.step()
    return running_loss


def valid_every_epoch(cfg, epoch, model, margin, loss_fn, valid_dataloader, device, scheduler=None, schd_batch_update=False, running_loss=True):
    model.eval()
    loss_sum = 0;
    sample_num = 0
    image_preds_all = []
    image_targets_all = []
    class_num = len(os.listdir(args.train_img_path))
    cmtx_all = torch.zeros(class_num, class_num, dtype=torch.float64)
    pbar = tqdm(enumerate(valid_dataloader), total=len(valid_dataloader))


    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()

        images_labels = image_labels.to(device).long()

        image_preds = model(imgs)
        # image_preds = margin(image_preds, image_labels)
        # print("image_preds:", image_preds)
        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()] #get index of max value along dim1
        image_targets_all += [images_labels.detach().cpu().numpy()]

        if args.cal_mtx == True:
            cmtx_current = calc_cmtx(image_preds, images_labels, class_num, reduce=None) #calculate cmtx every batch
            cmtx_all += cmtx_current


        loss_thisepoch = loss_fn(image_preds, images_labels)


        loss_sum += loss_thisepoch*image_labels.shape[0]
        sample_num += image_labels.shape[0]

        if ((step + 1) % cfg['verbose_step'] == 0) or ((step + 1) == len(valid_dataloader)):
            description = f'epoch {epoch} loss: { loss_sum/sample_num : .4f}'
            pbar.set_description(description)
    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    valid_acc = (image_preds_all == image_targets_all).mean()
    print('validation acc = {:.4f}'.format(valid_acc))
    if scheduler != None:
        if schd_batch_update:
            scheduler.step(loss_sum/sample_num)
        else:
            scheduler.step()

    return valid_acc, cmtx_all if args.cal_mtx else None

def train(train_csv, valid_csv, data_root_train, data_root_valid, classifier, modelarch, dataset=None, device=None):

    print('Training started')
    train_dataset = dataset(train_csv, data_root_train, transforms=get_train_transforms(CFG), output_label=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=CFG['train_batchsize'],
                                                   num_workers=CFG['num_workers'],
                                                   shuffle=True,
                                                   pin_memory=False)


    valid_dataset = dataset(valid_csv, data_root_valid, transforms=get_valid_transforms(CFG), output_label=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=CFG['valid_batchsize'],
                                                   num_workers=CFG['num_workers'],
                                                   shuffle=False,
                                                   pin_memory=False)

    class_num = len(os.listdir(data_root_train))
    model = classifier(modelarch, class_num, pretrained=True).to(device)
    # model = Net(model) #reduce the final layer

    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])

    #optimizer and scheduler
    # optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    # optimizer = torch.optim.SGD(model.parameters(), lr=CFG['lr'], momentum=0.9)



    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG['T_0'], T_mult=1, eta_min=CFG['min_lr'], last_epoch=-1)



    #define loss function
    if CFG['loss'] == 'labelsmoothCE':
        loss_train = CrossEntropyLabelSmooth(num_classes=len(os.listdir(args.train_img_path))).to(device)
        loss_valid = CrossEntropyLabelSmooth(num_classes=len(os.listdir(args.train_img_path))).to(device)
    elif CFG['loss'] == 'BCEwithlogits':
        loss_train = nn.BCEWithLogitsLoss().to(device)
        loss_valid = nn.BCEWithLogitsLoss().to(device)
    else:
        loss_train = nn.CrossEntropyLoss().to(device)
        loss_valid = nn.CrossEntropyLoss().to(device)

    #define margin type
    margin = ArcMarginProduct(in_feature=6, out_feature=len(os.listdir(args.train_img_path)), s=32, m=0.20)
    #args.feature_dim, trainset.class_nums, s=args.scale_size

    optimizer = torch.optim.SGD([
        {'params': model.parameters(), 'weight_decay': 5e-4},
        {'params': margin.parameters(), 'weight_decay': 5e-4}
    ], lr=0.1, momentum=0.9, nesterov=True)

    margin = margin.to(device)
    margin = torch.nn.DataParallel(margin, device_ids=[0,1,2,3])

    #train and valid
    best_acc = 0;
    for epoch in range(CFG['epochs']):
        loss = train_every_epoch(epoch, model, margin, loss_train, optimizer, train_dataloader, device, scheduler=None, schd_batch_update=False)
        writer.add_scalar('loss', loss, epoch, time.time())

        if (epoch+1) % CFG['valid_every_x_epoch'] == 0:
            with torch.no_grad():
                acc, cmtx = valid_every_epoch(CFG, epoch, model, margin, loss_valid, valid_dataloader, device, scheduler=None, schd_batch_update=False)
            writer.add_scalar('valid_acc', acc, epoch, time.time())

            #if current acc > old acc
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), '{}/{}_best_{:.4f}'.format(CFG['weights_path'], CFG['model_arch'], best_acc))
                torch.save(margin.state_dict(), '{}/{}_best_{:.4f}'.format(CFG['weights_path'], 'arcFaceModule', best_acc))
                if args.cal_mtx:
                    save_cmtx(cmtx, title=CFG['model_arch'], save_to_file=CFG['weights_path']+'cls_mtx' + '{:.4f}'.format(best_acc) +'.png')
            elif best_acc - acc <= 3:
                torch.save(model.state_dict(), '{}/{}_best_{:.4}'.format(CFG['weights_path'], CFG['model_arch'], acc))
                torch.save(margin.state_dict(), '{}/{}_best_{:.4f}'.format(CFG['weights_path'], 'arcFaceModule', acc))
                if args.cal_mtx:
                    save_cmtx(cmtx, title=CFG['model_arch'], save_to_file=CFG['weights_path']+'cls_mtx' + '{:.4f}'.format(acc) +'.png')
            #torch.save(model.state_dict(), '{}/{}_fold_{}_{}'.format(CFG['weights_path'], CFG['model_arch'], fold, epoch))

    del model, optimizer, train_dataloader, valid_dataloader #scheduler
    torch.cuda.empty_cache()

if __name__ == "__main__":
    #import config
    from config_smokingcalling import CFG

    if os.path.exists(CFG['weights_path']) is False:
        os.makedirs(CFG['weights_path'])


    #tensorboard wrtier
    writer = SummaryWriter(CFG['experiment_dir'])
    writer.add_text('training_setting', CFG['model_arch'])
    writer.add_text('training_setting', 'optimizer_SGD')
    writer.add_text('training_setting', 'batch_size:%s'%(CFG['train_batchsize']))
    writer.add_text('training_setting', 'init_lr"%f'%(CFG['lr']))
    #environ
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3' #set logic gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #make image data to csv
    train_csv = pd.read_csv(args.train_csv)
    valid_csv = pd.read_csv(args.valid_csv)
    if CFG['model_arch'][0:6] == 'resnet' or CFG['model_arch'][0:7] == 'resnest':
        classifier = Classifier
    elif CFG['model_arch'][0:3] == 'Rep': #repvgg
        classifier = Classifier_repvgg
    elif CFG['model_arch'][0:2] == 'tf':
        classifier = Classifier_efficientnet
    dataset = Classification_Dataset

    #model
    model = get_model(CFG["model_arch"])
    train(train_csv, valid_csv, args.train_img_path, args.valid_img_path, classifier, model, dataset, device)


    writer.close()
