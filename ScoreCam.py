from typing import Union
import logging

from PIL import Image
import torch

import numpy as np
import torch.nn.functional as F
import matplotlib.cm as mpl_color_map

from model.model_zoo import get_model
from model.models import Classifier
import matplotlib.pyplot as plt
import os
import copy
from model.models import Classifier_repvgg

from torchvision import transforms

def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL or ndarray img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # if isinstance(org_im, torch.Tensor):
    #     org_im = org_im.cpu().detach().numpy()
    # if isinstance(org_im, np.ndarray):
    #     if org_im.dtype in (np.float16, np.float32, np.float64):
    #         org_im = Image.fromarray((org_im * 255).astype(np.uint8))
    #     elif org_im.dtype == np.uint8:
    #         org_im = Image.fromarray(org_im)
    #     else:
    #         raise NotImplementedError(f"{org_im.dtype}")
    # else:
    #     raise NotImplementedError(f"{type(org_im)}")
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


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
    return im_as_ten


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model: torch.nn.Module, target_layer: Union[int, torch.nn.Module]):
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.conv_output = None #define specific layer output
        last_matched = None
        # register forward hook
        for idx, (n, m) in enumerate(model.named_modules()):
            if idx == target_layer or m is target_layer: #if match specific index
                if last_matched:
                    self.logger.warning(f"more than on submodules matched target_layer, "
                                        f"remove hook for last matched submodule: {last_matched[0]} and {n}")
                    last_matched[1].remove()
                last_matched = (n, m.register_forward_hook(hook=self.forward_pass_hook))
        assert last_matched is not None, "target layer not found."

    def forward_pass_hook(self, m, x, out): #define hook function,
        self.conv_output = out

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        x = self.model(x)
        assert self.conv_output is not None
        return self.conv_output, x


class ScoreCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None, target_size=(224, 224), sore_fn=torch.exp):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        input_image = input_image.to(device)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        print('model output:', model_output)
        need_target_class = False
        if target_class is None:
            need_target_class = True

            target_class = np.argmax(model_output.data.cpu().numpy()) #take the max value index of model output


        # Get convolution outputs
        target = conv_output[0]
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i in range(len(target)):
            # Unsqueeze to 4D
            saliency_map = torch.unsqueeze(torch.unsqueeze(target[i, :, :],0),0)
            # Upsampling to input size
            saliency_map = F.interpolate(saliency_map, size=target_size, mode='bilinear', align_corners=False)
            if saliency_map.max() == saliency_map.min():
                continue
            # Scale down between 0-1
            norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
            # Get the target score
            w = sore_fn(self.extractor.forward_pass(input_image*norm_saliency_map)[1])[0][target_class]
            cam += w.data.cpu().numpy() * target[i, :, :].data.cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-5)  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                                                    input_image.shape[3]), Image.ANTIALIAS))/255
        return cam, target_class if need_target_class else cam




if __name__ == '__main__':


    # if weights trained using dataparallel
    # if load weights trained using DataParallel, key is 'module.conv....'

    # pretrained_model = models.alexnet(pretrained=True)
    model = get_model('RepVGG-B1g2')
    # model = Classifier(model, 8, pretrained=False)
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

    classifier = Classifier_repvgg
    model = classifier(model, 8, pretrained=True).to(device)
    # print(model)


    from collections import OrderedDict

    weights = '/home/liwei.fang/classification_work/weights/20210305_removeworker_addcalling3/repvgg_b1g2_bs224_sgd/RepVGG-B1g2_best_0.5432'
    # weights = 'weights/20210114_head/resnest50_bs256_sgd/resnest50_best_0.9973'

    state_dict = torch.load(weights, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # model.load_state_dict(torch.load('weights/20210113_head_2/resnest50_bs256_sgd/resnest50_best_0.9979', map_location=device))
    # model.to(device)


    from torchvision import models
    # for imgs in os.listdir("data_inference/spc_head_test_20210112_2/smoking/"):
    #     ori_img = Image.open("data_inference/spc_head_test_20210112_2/smoking/" + imgs)
    #     ori_img = ori_img.resize((224, 224))
    #     prep_img = preprocess_image(ori_img)
    #     score_cam = ScoreCam(model, target_layer=model.model.layer4[2])
    #     cam, target_cls = score_cam.generate_cam(prep_img)
    #     heatmap, heatmap_on_image = apply_colormap_on_image(ori_img, cam, "jet")
    #     # plt.title(str(target_cls), fontsize=16)
    #     if target_cls == 2:
    #         plt.imshow(heatmap_on_image)
    #         plt.show()
    #

    label = ['normal', 'normal2', 'normal3', 'smoking', 'calling', 'calling2', 'calling3', 'calling4']
    #loading one img
    # ori_img = Image.open('/home/liwei.fang/classification_work/data_inference/spc_head_test_20210112_2/smoking/9067.jpg')
    ori_img = Image.open('/home/liwei.fang/classification_work/data_inference/spc_head_test_20210305_remove_worker_addcalling3/normal2/20210301_3_live_1592_0.91.jpg')
    # ori_img = Image.open('/home/liwei.fang/classification_work/data/spc_smokingcalling_20210113_2/calling/3587.jpg')
    ori_img = ori_img.resize((224, 224))
    prep_img = preprocess_image(ori_img)
    # score_cam = ScoreCam(model, target_layer=model.model.layer4[2])
    score_cam = ScoreCam(model, target_layer=model.model.stage4[0])
    cam, target_cls = score_cam.generate_cam(prep_img)
    print('target class:%s' %(label[target_cls]))
    heatmap, heatmap_on_image = apply_colormap_on_image(ori_img, cam, "jet")
    heatmap_on_image.save('20210301_3_live_1592_0.91_output_'+label[target_cls]+'.png')
    # plt.imshow(heatmap_on_image)
    # plt.show()