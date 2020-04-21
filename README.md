# Image_classification_Pytorch
本来是自己写来比赛用， 方便调参的图像分类框架, 一个挺方便分类框架， 还在慢慢完善各种模型

目前包含

- Resnet18
- Resnet50
- Se_resnext50
- EfficientNetB0
- EfficientNetB5
- Resnest50
- Resnest101



#### install

```shell
git clone https://github.com/Stephenfang51/image_classification_pytorch
```



#### requirement

1. torch 版本必须 >= 1.1 而且必须从 pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl 这样安装 cuda才会是10.0

2. `pip install pretrainedmodels`



#### train data

1. Preparing your train data like 

   ```
   # if you have 3 classes in total, they are class A, class B, class C
   # your train will be like below
   
   train/A/
   train/B/
   train/C/
   ```

2. if your training image all in one dir, you can use [csv_data_classification.ipynb](https://github.com/Stephenfang51/image_classification_pytorch/blob/master/csv_data_classification.ipynb) to move your original imgs to its independant dir according to your csv file

#### How to train

you can directly command like below to train, select your model, input_size, num_classes …etc

```shell
python train.py --train_path /input/mango/train_3levels/ --model efficientb5 --input_size 224 --num_classes 3 --lr_scheduler steplr --step_size 20 --batch_size 32 --init_lr 0.001 --freeze True
```

To see all the command param, you can check [train_detail.py](https://github.com/Stephenfang51/image_classification_pytorch/blob/master/train_detail.py)



#### Update

>update 2020.4.21

1. 新增ResNest50, ResNest101

   - model/utils.py  freeze_param, select_network添加ResNest50, ResNest101

   - Train_detail —model 添加ResNest50, ResNest101

2. utils.py 下opcounter 修复格式

   

> update 2020.3.21

1. 改一下train.py (freeze的部分)
2. 将freeze_param添加到model/utils.py底下
3. data_aug 修正 RandomErasing , 所以dataset.py也修正
4. train.py / train_detail.py 也修正

>update 2020.3.20

1. dataset.py 修改 (待测试randomErasing)
2. train.py修改（新增efficientNet, 增加steplr scheduler)
3. train_detail.py (增加steplr scheduler)
4. 新增models/utils.py (select_networks)
5. 修改utils.py (新增select_scheduler, opcounter)


>update 2020.3.16

1. 让argparser support bool值当命令行参数
2. 添加resume/checkpoint参数， resume表示要从checkpoint继续训练， checkpoint 选择要续run的model name， 默认从`./checkpoints/`找
3. 更新test.py 新增test_detail.py吗， 主要完成读取模型并且inference后生成csv档
4. 更新dataset中的CDC类， 将test_path传入

>update 2020.3.13

组建一个图像分类通用框架， 用torch搭建

待完善

1.test.py
2.model.py (各种常用分类模型）
3.requirement 文件

