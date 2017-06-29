close all;clear;clc;
%网络定义
cnn.layers = {
    struct('type', 'input') %input layer
    struct('type', 'conv', 'featuremaps', 6, 'kernelsize', [5,5], 'stride', 1) %convolution layer
    struct('type', 'actfun','function','relu') %activation function layer
    struct('type', 'bn') %batch-normalization layer
    struct('type', 'pool', 'kernelsize', 2, 'method', 'max','weight',0) %pool layer
    struct('type', 'conv', 'featuremaps', 16, 'kernelsize', [3,3], 'stride', 1) %convolution layer
    struct('type', 'actfun','function','relu') %activation function layer
    struct('type', 'bn') %batch-normalization layer
    struct('type', 'conv', 'featuremaps', 48, 'kernelsize', [3,3], 'stride', 1) %convolution layer
    struct('type', 'actfun','function','relu') %activation function layer
    struct('type', 'pool', 'kernelsize', 2, 'method', 'mean','weight',0) %pool layer
    struct('type', 'bn') %batch-normalization layer
    struct('type', 'conv', 'featuremaps', 120, 'kernelsize', [2,2], 'stride', 1) %convolution layer
    struct('type', 'actfun','function','relu') %activation function layer
    struct('type', 'bn') %batch-normalization layer
    struct('type', 'fc', 'featuremaps', 64) %full connecting layer
    struct('type', 'actfun','function','relu') %activation function layer
    struct('type', 'bn') %batch-normalization layer
    struct('type', 'fc', 'featuremaps', 10) %full connecting layer
    struct('type', 'loss','function', 'softmax') %loss layer
    };
%训练参数定义
opts.alpha = 0.5;   %初始学习率
opts.momentum = 0.5;  %初始动量项权值
opts.batchnum = 1000; %批大小
opts.numepochs = 10;  %迭代次数

%导入数据和类标(one vs all)
load mnist_uint8.mat; %加载数据集 
train_data = double(reshape(train_x',28,28,60000))/255;
test_data = double(reshape(test_x',28,28,10000))/255;
train_label = double(train_y');
test_label = double(test_y');
numClasses = 10;
[height,width,datanum] = size(train_data);
batchsize = [height,width,opts.batchnum];
% 建立CNN网络
inputSize = batchsize; %输入图片大小
outputSize = numClasses; %输出类别数目
cnn= nn_setup(cnn, inputSize, outputSize);
% 训练网络
cnn = nn_train(cnn,train_data,train_label,opts);
% 测试网络
[accuracy, index] = nn_test(cnn,test_data,test_label);
