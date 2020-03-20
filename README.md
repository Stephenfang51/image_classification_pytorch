# image_classification

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

