import random
import math
import numpy as np
from PIL import Image


def RandomErasing(im, probability=0.5, sl=0.02, sh=0.4, rl=0.3, mean=[128, 128, 128]):
    # -----------几率大于50% 则不对图像改变----------#
    if random.uniform(0, 1) > probability:
        return im

    else:
        img = np.array(im)
        area = img.shape[0] * img.shape[1]

        while True:
            # ------------RE的区域---—------------------#
            target_area = random.uniform(sl, sh) * area
            aspect_ratio = random.uniform(rl, 1 / rl)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if img.shape[0] > w and img.shape[1] > h:
                break
            # 如果目標區域沒超過原圖， 則定義目標矩形x1, y1起始點
        if w < img.shape[0] and h < img.shape[1]:
            x1 = random.randint(0, img.shape[0] - w)
            y1 = random.randint(0, img.shape[1] - h)
            # --確認如果是三層RGB圖像，每一層的目標矩形加上像素值128----#
            if im.mode == 'RGB':
                img[x1:x1 + h, y1:y1 + w, 0] = mean[0]
                img[x1:x1 + h, y1:y1 + w, 1] = mean[1]
                img[x1:x1 + h, y1:y1 + w, 2] = mean[2]
            elif im.mode == 'L':
                img[x1:x1 + h, y1:y1 + w] = mean[0]
        img = Image.fromarray(np.uint8(img))

        return img