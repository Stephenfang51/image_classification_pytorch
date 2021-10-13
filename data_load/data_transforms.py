from albumentations import (    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize)


from albumentations.pytorch import ToTensorV2

def get_train_transforms(cfg):
    return Compose([
                    # RandomResizedCrop(cfg['img_size'], cfg['img_size']),
                    Resize(cfg['img_size'], cfg['img_size']),
                    Transpose(p=0.5),
                    HorizontalFlip(p=0.5),
                    VerticalFlip(p=0.5),
                    ShiftScaleRotate(p=0.5),
                    HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                    RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                    CoarseDropout(p=0.5),
                    Cutout(p=0.5),
                    ToTensorV2(p=1.0),
                    ], p=1.)

def get_valid_transforms(cfg):
    return Compose([
            # CenterCrop(cfg['img_size'], cfg['img_size'], p=1.),
            Resize(cfg['img_size'], cfg['img_size']),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)


def get_inference_transforms(cfg):
    return Compose([
            Resize(cfg['img_size'], cfg['img_size']),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)


def get_test_tta_transforms(cfg):
    return Compose([
            # RandomResizedCrop(cfg['img_size'], cfg['img_size']),
            Resize(cfg['img_size'], cfg['img_size']),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)