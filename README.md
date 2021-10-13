#### Update



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

1. ResNet models
2. ResNest models
3. efficientNet B4

---

### How to train

edit **config.py** to change hyper-parameters 

`weights_path` is to save the best weights

`experiment_dir` is to save tensorboard log

`class_num` : class number

```
CFG = {
    'fold_num': 5,
    'seed': 719,
    'model_arch': 'tf_efficientnet_b4_ns',
    'img_size': 224,
    'epochs': 100,
    'train_batchsize': 32,
    'valid_batchsize': 128,
    'T_0': 10,  #for cosine lr scheduler
    'lr': 1e-3,
    'min_lr': 1e-6,
    'weight_decay':1e-6,
    'num_workers': 4,
    'accum_iter': 2, # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 1,
    'device': 'cuda:0',
    'weights_path' : 'weights/efficientb4_bs32_sgd',
    'experiment_dir': 'runs/efficientb4_sgd/',
    'class_num' : 3

}
```





and you can decide not to using pretrained model in `train.py`, under train function

and finally just `python train.py --img_path /path/to/data/ --csv /path/to/xxx.csv`



#### train with different validation

use `train_.py`

should specifiy `valid_img_path`, `valid_csv` as arguments

---

#### Inference 

#### test img one by one

create a dir containing a list of imgs

and using `test.py` 

e.g.

`python test.py --img_path /path/to/inference --save_path /path/to/saveresult --model 'enteryour model_arch --num_class 'enter your num of class' --weights enter your checkpoint path`

and predicted class will be draw on output imgs

#### test img with TTA (test-time augmentation)

using `test_tta.py`

specify your 

1. pretrained model 
2. test data path 
3. csv



---

#### ScoreCam

edit `ScoreCam.py` 

```python
ori_img = Image.open('photo/smoking.jpg').convert('RGB')
```

and run on pycharm to directly show image using matplotlib