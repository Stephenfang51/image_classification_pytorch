#### Update


2021.10.13
1. reconstruct whole code structure


2021.2.6

1. add labelsmooth flag in `config.py` and `train.py` and `utils/loss.py`

2021.1.7

1. `preprocess.py` add new column now `class_name`
2. add `test_tta.py` (not finished), `train_.py`(support different validation) 

2021.1.5

1. add `ScoreCam.py`to visualize CNN layer
2. add confusion_matrix to `train.py`, add new argument `--cal_mtx` default is `True`

2021.1.4

1. modified the `model/models`, to change the `Classifier`
2. modified the `train.py` 
3. modified the `test.py`

2020.12.31 

1. update preprocess_data for easily produce csv file
2. update test.py for easily doing inference

2020.12.29 upload to github

#### supported models

2020.12.29

1. ResNet all models
2. ResNest all models
3. efficientNet B4

---

## How to train

### 1. data prepare
1. set you image data be like :
```
data
  | --- task_name
            |---train
                  |---class1
                        |---- img1.jpg
                        |---- img2.jpg
                        |---- ....
                  |---class2
                  |---...
                    
            |--- val
                  |---class1
                        |---- img1.jpg
                        |---- img2.jpg
                        |---- ....
                  |---class2
                  |---...
```

2.using data_load/img2csv.py to create train and val csv file
```shell
#create train csv
python data_load/img2csv.py --img_path data/your_task_name/train/ --classes class1,class2... --csv data/your_task_name_train.csv
#create vaL csv
python data_load/img2csv.py --img_path data/your_task_name/ --classes class1,class2... --csv data/your_task_name_val.csv
```


### 2. start training

you can copy a default cfg to modify it in experiments dir:

config.yaml
```
fold_num: 5
seed: 719
model_arch: resnet50
img_size: 224 
epochs: 100
loss: labelsmooth  #BCEwithlogits, labelsmooth, c
train_batchsize: 256
valid_batchsize: 128
test_batchsize: 128 #for test tt
T_0: 10  #for cosine lr schedule
lr: 0.001
min_lr: 0.00001
weight_decay: 0.00001
num_workers: 4
accum_iter: 2 # suppoprt to do batch accumulation for backprop with effectively larger batch siz
verbose_step: 1
tta : 3
valid_every_x_epoch: 5
default_save_path: output_model
gpus : 0
# 'model_arch': "tf_efficientnet_b4_ns"
# 'model_arch': 'RepVGG-B1g2'
```

train commnad example
```shell
python train.py --tpath data/yourtask/train/ --tcsv data/yourtask_train.csv --vpath data/yourtask/val/ --vcsv data/yourtask_val.csv --cfg experiment/config_task.yaml --bsize 128 --gpus '0,1,2,3,4,5,6,7'
```
- `-bsize` : training-batchsize
- `-cfg` : your config yaml
- `-gpus` : set gpu be like '0,1,2,3' ps.start from 0

### 3. after training
1. training confusion matrix 
2. trained weight
3. tensorboard log

above will be save into output_model directory by date_modelname

---

### Inference 

not yet update


---

#### ScoreCam

not yet update