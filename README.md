# DogClassification
# 原理说明

首先，将数据按照mat文件所记录的方式分为训练集和测试集两部分。由于12000张的训练集有点少，所以对每张训练图片采集它的左上角80%区域、右上角80%、左下角80%、右下角80%、中间80%，然后对其进行水平翻转，这样一张图片就变成了10张。

接着，训练方式采用迁移学习，使用已经训好的VGG16模型，去掉最后几层的全连接层后，接上自己的模型进行训练。每张图片丢进VGG16的卷积层得到一个`7*7*512`维的特征向量，flatten之后，由于数据量少，将后续的两个全连接层的维度从4096降到1024，dropout取0.5（防止过拟合），最后加上softmax，类别设置为120。

# 训练结果

```
Epoch 500/500
120000/120000 [==============================] - 180s - loss: 3.8973 - acc: 0.7559
```

```
				precision    recall  f1-score   support
avg / total       0.49      0.58      0.52      8580
```

# 关于数据
使用的是[Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)数据集，类别有120类，图片共有20580张，其中训练集12000张，测试集8580张。需要下载的部分包括Images（757MB的图片压缩包）、Train Features和Test Features（1.2GB+850MB的matlab文件，用于把数据分为训练集和测试集合，不过好像下载Lists这个0.5MB的压缩包就可以了，不过在main.py中的代码需要更改下）

> 运行前需要将代码中的数据位置进行更改

# 依赖环境

* Anconda3 (3.6.0 64-bit)
* Keras (2.0.5)
* tensorflow-gpu (1.0.1)（keras后端使用的是tensorflow）
* opencv (3.1.0)
* pillow (3.4.3)
* scipy (0.19.1)
* numpy (1.12.1)



# 使用方法
改好数据的路径后，直接执行`python main.py`即可

# TODO

F1-score目前还只有52%，等这阵子忙好了，再找论文看看