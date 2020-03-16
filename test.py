from  torch.utils.data import DataLoader
from torchvision import transforms
from test_detail import test_detail
from dataset import CDCDataset
import pandas as pd
import torch
import os


#csv and mode


test_opt = test_detail().parse()

Train_path = test_opt.train_path #为了要生成cls2label， 考虑用别的方法
Size = test_opt.input_size
Batch_size = test_opt.batch_size

Csv = test_opt.csv
Pre_model =test_opt.pre_model
Test_path = test_opt.test_path



df_test = pd.read_csv(Csv) #读取csv档案
test_data = df_test['Id']

cls2label = {}
for label, cls in enumerate(os.listdir(Train_path)):
    cls2label[cls] = label
print(cls2label)

def inference(model):
    test_transform = transforms.Compose([transforms.Resize((int(Size), int(Size))),
                                         # transforms.TenCrop(Size),
                                         # Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])

    dst_test = CDCDataset(test_data, transform=test_transform, mode='test', test_path = Test_path)
    dataloader_test = DataLoader(dst_test, shuffle=False, batch_size=Batch_size // 2, num_workers=8)

    model.eval()
    results = []
    print('start inferencing')
    for ims, im_names in dataloader_test:
        input = ims.requires_grad_().cuda()
        output = model(input)
        _, preds = output.topk(1, 1, True, True)  # 取top 1
        preds = preds.cpu().detach().numpy()
        for pred, im_name in zip(preds, im_names):
            top1_name = [list(cls2label.keys())  # 字典取key（cls)
                         [list(cls2label.values()).index(p)]  # 字典取value(label)
                         for p in pred]
            results.append({'Id': im_name, 'pred_cls': ''.join(top1_name)})
    df = pd.DataFrame(results, columns=['pred_cls', 'Id'])
    df.to_csv('sub.csv', index=False)


if __name__ == '__main__':
    model = torch.load(os.path.join('./checkpoints/{}.pth'.format(Pre_model))) #读取整个模型
    inference(model)