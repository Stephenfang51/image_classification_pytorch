import os
import torch
import cv2
import argparse
import time
import numpy as np
import torch.nn as nn
from model.model_zoo import get_model
from config import CFG
from data_load.data import get_image
from data_load.data_transforms import get_inference_transforms
from model.models import Classifier_efficientnet, Classifier
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


"""
python test.py --img_path photo/ --save_path test_results/ --model resnest50 --num_class 3 --weights weights/20210113_head_2/resnest50_bs256_sgd/resnest50_best_0.9979 
"""

"""---options---"""
parser = argparse.ArgumentParser(description='inference testing')
parser.add_argument('--img_path', type=str, required=True, help='enter the image main path')
parser.add_argument('--save_path', type=str, required=True, help='save result image')
parser.add_argument('--model', type=str, required=True, help='which model to test, e.g. resnet18')
parser.add_argument('--num_class', type=int, required=True, help='the num of classes')
parser.add_argument('--weights', type=str, required=True, help='which weights to load')

args = parser.parse_args()

#remember to record time takes


# load model
# load tf efficientNet
if args.model[0:2] == 'tf':
    model = Classifier_efficientnet(args.model, args.num_class, False)
else:
    # load other models e.g. resnet or resnest
    model = get_model(args.model)
    model = Classifier(model, args.num_class, False)

# load pretrained model
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model.load_state_dict(torch.load(args.weights, map_location=device))
# model.to(device)
# model.eval()


#if weights trained using dataparallel
# if load weights trained using DataParallel, key is 'module.conv....'
from collections import OrderedDict
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
state_dict = torch.load(args.weights, map_location=device)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.to(device)
model.eval()

def preprocess_image(pil_im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): PIL Image or numpy array to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    #ensure or transform incoming image to PIL image
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print("could not transform PIL_img to a PIL Image object. Please check input.")

    # Resize image
    if resize_im:
        pil_im = pil_im.resize((224, 224), Image.ANTIALIAS)

    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    im_as_ten.requires_grad_()
    im_as_ten = im_as_ten.to(device)
    return im_as_ten

def test(class_name):
    #start inference
    with torch.no_grad():
        timetable = np.array([])
        for img in os.listdir(args.img_path):
            start = time.time()

            img_name = str(img)
            each_img_path = args.img_path + img
            # src_img = cv2.imread(each_img_path) #copy img wait to be put Text
            # img = get_image(each_img_path)
            # img_transformer = get_inference_transforms(CFG)
            # ori_img = Image.open('/home/liwei.fang/classification_work/photo/20210113_smoking_1.jpg')
            ori_img = Image.open(each_img_path)
            src_img = ori_img
            ori_img = ori_img.resize((224, 224))
            img = preprocess_image(ori_img)


            # img = img_transformer(image=img)['image']
            # img = img.to(device).float()
            # img = torch.unsqueeze(img, 0) #batch size = 1
            image_pred = model(img)
            image_pred = image_pred.view((1, 3))

            softmax = torch.nn.Softmax()
            output = softmax(image_pred)
            # topk = output.topk(1)[1]
            end = time.time()
            timetable = np.append(timetable, end-start)
            # topk = topk.cpu().numpy()

            #if take top one must use topk[0][0]
            img_draw = ImageDraw.Draw(src_img)
            y0, dy = 20, 20
            for i, class_txt in enumerate('n:%.5f \n s:%.5f \n c:%.5f'.split('\n')):
                y = y0 + i * dy
                img_draw.text((2, y), class_txt %(output[0][i]), fill=(0,255,0))
            src_img.save(args.save_path+img_name)
            # output_img = cv2.putText(src_img, class_txt %(output[0][i]), (2, y), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 0), 1)
        # cv2.imwrite(args.save_path + img_name, output_img)
    print('average inference time:%.3f s' %(timetable.mean()))

if __name__ == "__main__":
    if os.path.exists(args.save_path) != True:
        os.mkdir(args.save_path)


    # model = get_model(args.model)
    class_names = ['normal', 'smoking', 'calling']

    test(class_name=class_names)
