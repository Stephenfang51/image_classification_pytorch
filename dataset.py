import os
from torch.utils.data import Dataset
from PIL import Image



class CDCDataset(Dataset):
    def __init__(self, dataset, transform=None, mode='train', tta=False, idx=0):
        self.tta=tta
        self.idx = idx
        self.mode = mode
        if self.mode == 'train' or self.mode == 'val':
            self.ims, self.labels = [], []
            for item in dataset.items():
                self.ims.append(item[0])
                self.labels.append(item[1])
            # print(self.ims, self.labels)
        elif self.mode == 'test':
            self.im_names = dataset
            self.ims = [os.path.join('../input/test/test/0', im) for im in dataset]
        self.transform = transform

    def __getitem__(self, index):
        im_path = self.ims[index]
        if self.mode == 'train' or self.mode == 'val':
            label = self.labels[index]
        elif self.mode == 'test':
            im_name = self.im_names[index]
        im = Image.open(im_path)
        if self.mode == 'train' or self.mode == 'val':
            if self.transform is not None:
#                 im = RandomErasing(im, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[128, 128, 128])
                im = self.transform(im)
            return im, label
        elif self.mode == 'test':
            if self.tta:
                w, h = im.size
                if self.idx == 0:
                    im = im.crop((0, 0, int(w*0.9), int(h*0.9))) # top left
                elif self.idx == 1:
                    im = im.crop((int(w*0.1), 0, w, int(h*0.9))) # top right
                elif self.idx == 2:
                    im = im.crop((int(w*0.05), int(h*0.05), w-int(w*0.05), h-int(h*0.05))) # center
                elif self.idx == 3:
                    im = im.crop((0, int(h*0.1), w-int(w*0.1), h)) # bottom left
                elif self.idx == 4:
                    im = im.crop((int(w*0.1), int(h*0.1), w, h)) # bottom right
                elif self.idx == 5:
                    im = im.crop((0, 0, int(w*0.9), int(h*0.9)))
                    im = im.transpose(Image.FLIP_LEFT_RIGHT) # top left and HFlip
                elif self.idx == 6:
                    im = im.crop((int(w*0.1), 0, w, int(h*0.9)))
                    im = im.transpose(Image.FLIP_LEFT_RIGHT) # top right and HFlip
                elif self.idx == 7:
                    im = im.crop((int(w*0.05), int(h*0.05), w-int(w*0.05), h-int(h*0.05)))
                    im = im.transpose(Image.FLIP_LEFT_RIGHT) # center and HFlip
                elif self.idx == 8:
                    im = im.crop((0, int(h*0.1), w-int(w*0.1), h))
                    im = im.transpose(Image.FLIP_LEFT_RIGHT) # bottom left and HFlip
                elif self.idx == 9:
                    im = im.crop((int(w*0.1), int(h*0.1), w, h))
                    im = im.transpose(Image.FLIP_LEFT_RIGHT) # bottom right and HFlip
            if self.transform is not None:
                im = self.transform(im)
            return im, im_name

    def __len__(self):
        return len(self.ims)