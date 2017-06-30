# CN4Mat
Convolution Neural Networks for Matlab

# 1 简介

    此工具箱适用于Matlab环境下卷积神经网络的构造和学习，参考了rasmusbergpalm的DeepLearnToolbox[1]，并借鉴了 caffe的一些思想。相比于原始的CNN框架，添加了许多新的网络结构，适合新手入门，了解各个构件的连接方式、前向传播和反向传播的计算过程，方便以后造轮子~~
    此工具箱只是有助于你熟悉卷积神经网络和深度学习，鉴于Matlab的效率，不建议使用此工具箱训练较大的模型（当然，你也可以尝试并行），建议使用现在比较流行的深度学习框架:

【Caffe】http://caffe.berkeleyvision.org/

【TensorFlow】http://tensorflow.org

【Theano】http://deeplearning.net/software/theano/ 

【Mxnet】http://mxnet.io

【Torch】http://torch.ch

【Keras】https://keras.io/

【MatConvNet】http://www.vlfeat.org/matconvnet/

# 2 使用方法

## 2.1 卷积层 

|parameters |arguments
|'type'     |类型，'conv'
|'featuremaps'  |特征数目（即卷积核数目），如 10
|'kernelsize'   |卷积核尺寸，如3（方形），[3,4]（矩形）
'stride'    |步长尺寸，如2（方形），[2,1]（矩形），默认为[1,1]
|'pad'  |外部填充尺寸，如2（方形），[2,1]（矩形），默认为[0,0]
注：卷积层只是卷积操作，没有激活函数

2.2 转置卷积层

'type' 	类型，'deconv'

'featuremaps'	特征数目（即卷积核数目），如 10

'kernelsize'	卷积核尺寸，如3（方形），[3,4]（矩形）

'stride'	步长尺寸，如2（方形），[2,1]（矩形），默认为[1,1]

'pad'	外部填充尺寸，如2（方形），[2,1]（矩形），默认为[0,0]

注：转置卷积层只是转置卷积操作，没有激活函数

2.3 池化层

'type' 	类型，'pool'

'featuremaps'	无需定义，默认和上一层相同

'kernelsize'	池化尺寸，如3（方形），[3,4]（矩形）

‘method’	池化方式，‘max’或‘mean’

‘weight’	池化层是否有权值，没有（0，默认），有（1）

'stride'	无需定义，默认同池化尺寸

'pad'	不支持

注：池化层后面不支持继续接一个池化层

2.4 Batch Normalization层

'type' 	类型，'bn'

'featuremaps'	无需定义，默认和上一层相同

注：BN层后面不支持继续接一个BN层

2.5 激活函数层

'type' 	类型，'actfun'

'featuremaps'	无需定义，默认和上一层相同

‘function’	激活函数，可选‘softmax’，‘tanh’，‘relu’

注：激活函数层后面不支持继续接一个激活函数层

2.6 全连接层

'type' 	类型，'fc'

'featuremaps'	特征数目（即卷积核数目），如 10

注：全连接层之前一层若是二维输出，需要矢量化

2.7 损失函数层

'type' 	类型，'loss'

'featuremaps'	无需定义，默认和上一层相同

‘function’	传递函数，可选‘softmax’，‘tanh’，‘relu’，‘softmax’

注：损失层之前必须接一个全连接层，且特征数目等于类别数目（one vs all）

# 3 版本更新

时间	更新内容

2016.12	第一版，在DeepLearnToolbox上增加了矩形卷积核、max pooling，tanh传递函数，relu传递函数，softmax损失函数，非单位卷积步长，非2倍池化

2017.5	添加 Batch Normalization层

2017.6	第二版，模块化改写，添加转置卷积层，周围填充尺寸map，矩形池化尺寸，矩形步长

待优化：

1. 添加权重衰减项
2. 添加Dropout层
3. 并行化

# 4 Bug

现阶段，所有层都已通过数值梯度检查，欢迎大家举报bug，联系我：
vivi_liu65@163.com 

# 5 参考

[1] https://github.com/rasmusbergpalm/DeepLearnToolbox

[2] Notes on Convolutional Neural Networks [Bouvrie, 2006]

[3] A guide to convolution arithmetic for deep learning [Vincent, 2016]

[4] Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift [Sergey, 2015]
