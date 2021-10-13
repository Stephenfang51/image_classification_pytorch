from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, SequentialSampler, RandomSampler
from sklearn.model_selection import GroupKFold, StratifiedKFold
import cv2
from .data_transforms import get_train_transforms, get_valid_transforms, get_test_tta_transforms



class Classification_Dataset(Dataset):
    def __init__(self, df, data_root, transforms = None, output_label=True):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.output_label = output_label

    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, index : int):
        if self.output_label:
            target = self.df.iloc[index]['label']

        label = self.df.iloc[index]['label']
        # path = "{}/{}".format(self.data_root, self.df.iloc[index]['id'])
        path = '{}{}/{}'.format(self.data_root, self.df.iloc[index]['class_name'], self.df.iloc[index]['id'])
        img = get_image(path)
        if self.transforms:
            img = self.transforms(image=img)['image']
        #for label smoothing
        if self.output_label == True:
            return img, target
        else:
            return img


#dataset_fuc = SmokingCalling_Dataset()
def prepare_dataloader(cfg, df, train_idx, valid_idx, data_root, dataset_func):
    train_ = df.loc[train_idx, :].reset_index(drop=True)
    valid_ = df.loc[valid_idx, :].reset_index(drop=True)

    #create dataset first and DataLoader to load
    train_dataset = dataset_func(train_, data_root, transforms=get_train_transforms(cfg), output_label=True)
    valid_dataset = dataset_func(valid_, data_root, transforms=get_valid_transforms(cfg), output_label=True)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg['train_batchsize'],
                                  pin_memory=False,
                                  drop_last=False,
                                  shuffle=True,
                                  num_workers=cfg['num_workers'],
                                  )
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg['valid_batchsize'],
                                  pin_memory=False,
                                  shuffle=False,
                                  num_workers=cfg['num_workers'],
                                  )
    return train_dataloader, valid_dataloader

def prepare_test_dataloader(cfg, df, test_idx, data_root, dataset_func):
    test_ = df.loc[test_idx, :].reset_index(drop=True)
    test_dataset = dataset_func(test_, data_root, transforms=get_test_tta_transforms(cfg), output_label=True)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg['test_batchsize'],
                                  pin_memory=False,
                                  shuffle=False,#create dataset first and DataLoader to load
                                  num_workers=cfg['num_workers'],
                                  )
    return test_dataloader




def get_image(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    return im_rgb

