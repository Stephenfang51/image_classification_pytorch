import torch
import pandas as pd
import os
import numpy as np
import argparse
from tqdm import tqdm
from data_load.data import Classification_Dataset
from model.models import Classifier
from model.models import Classifier_efficientnet
from model.model_zoo import get_model
from utils.fusion_matrix import calc_cmtx, save_cmtx
from data_load.data_transforms import get_test_tta_transforms

"""---options---"""
parser = argparse.ArgumentParser(description='Test with TTA')
parser.add_argument('--img_path', type=str, required=True, help='training data path')
parser.add_argument('--weights', type=str, required=True, help='which weights to load')
parser.add_argument('--csv', type=str, required=True, help='training csv file to load')
parser.add_argument('--cal_mtx', type=bool, default=True, help="whether to calculate fusion matrix")

args = parser.parse_args()


def test_every_epoch(model, test_dataloader, device):
    model.eval()
    image_preds_all = []
    image_targets_all = []
    cmtx_all = torch.zeros(CFG['class_num'], CFG['class_num'], dtype=torch.float64)
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))


    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        images_labels = image_labels.to(device).long()
        image_preds = model(imgs)
        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [images_labels.detach().cpu().numpy()]

        if args.cal_mtx == True:
            cmtx_current = calc_cmtx(image_preds, images_labels, CFG["class_num"], reduce=None)
            cmtx_all += cmtx_current
    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    test_acc = (image_preds_all == image_targets_all).mean()
    print('test acc = {:.4f}'.format(test_acc))

    return test_acc, cmtx_all if args.cal_mtx else None

def test_tta(test_csv, data_root, classifier, modelarch, dataset):
    # folds = StratifiedKFold(n_splits=CFG['fold_num'])
    # folds = StratifiedKFold(n_splits=1)
    # folds = folds.split(np.arange(test_csv.shape[0]), test_csv.label.values) #np.arange from shape0 to
    # for fold, (test_idx) in enumerate(folds):
    #     if fold > 0:
    #         break

    print('Test TTA with {} started')
    # test_ = test_csv.loc[test_idx, :].reset_index(drop=True)
    test_dataset = dataset(test_csv, data_root, transforms=get_test_tta_transforms(CFG), output_label=True)

    #test_dataloader
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=CFG['test_batchsize'],
                                                  num_workers=CFG['num_workers'],
                                                  shuffle=False,
                                                  pin_memory=False)
    model = classifier(modelarch, CFG['class_num'], pretrained=True).to(device)
    test_preds = []


    #if load weights trained using DataParallel, key is 'module.conv....'
    state_dict = torch.load(args.weights, map_location=device)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    for i, epoch in enumerate([6,7,8,9]): #4

        # model.load_state_dict(torch.load(args.weights, map_location=device))
        model.load_state_dict(new_state_dict)

        with torch.no_grad():
            for _ in range(CFG['tta']): #3
                acc, cmtx = test_every_epoch(model, test_dataloader, device)
                test_preds += [1/4/CFG['tta']*acc]
                print(test_preds)
                if args.cal_mtx:
                    weight_name = args.weights.split('/')[-1]
                    save_cmtx(cmtx, title=CFG['model_arch'],
                              save_to_file=(args.weights.strip(weight_name)) + 'test' + "_cls_mtx" + '{:.4f}'.format(acc) + '.png')



    # test_preds = np.mean(test_preds, axis=0)
    # print('test acc = {:.5f}'.format(test_csv.label.values==np.argmax(test_preds, axis=0).mean()))


    del model, test_dataloader
    torch.cuda.empty_cache()

if __name__ == "__main__":
    #import config
    from config import CFG
    #environ
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3' #set logic gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #loading test csv file
    test_csv = pd.read_csv(args.csv)

    if CFG['model_arch'][0:6] == 'resnet' or CFG['model_arch'][0:7] == 'resnest':
        classifier = Classifier
    else:
        classifier = Classifier_efficientnet
    dataset = Classification_Dataset




    #model
    model = get_model(CFG["model_arch"])
    test_tta(test_csv, args.img_path, classifier, model, dataset)
