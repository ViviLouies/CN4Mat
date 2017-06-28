%%网络的数值梯度核对
clear;clc;
%网络定义
cnn.layers = {
    struct('type', 'input') %input layer
    struct('type', 'conv', 'featuremaps', 2, 'kernelsize', [2,3], 'stride', 2, 'pad', 1) %convolution layer
    %struct('type', 'bn') %bn layer
    %struct('type', 'actfun','function','sigmoid') %actfun layer
    %struct('type', 'bn') %bn layer
    struct('type', 'pool', 'kernelsize',2, 'method', 'mean','weight',1) %pool layer
    struct('type', 'bn') %bn layer
    struct('type', 'deconv', 'featuremaps', 4, 'kernelsize', 2, 'stride',1 , 'pad', 0) %transpose convolution layer
    %struct('type', 'bn') %bn layer
    struct('type', 'fc', 'featuremaps', 2) %full connecting layer
    %struct('type', 'bn') %bn layer
    struct('type', 'actfun','function','tanh') %bn layer
    struct('type', 'bn') %bn layer
    struct('type', 'fc', 'featuremaps',4) %full connecting layer
    struct('type', 'loss','function','softmax') %loss layer
    };
%注意：如果卷积层使用的是relu激活函数（在0点不可导！！），则在计算卷积层的数值梯度时，可能不准确
%若x<0,则f(x)=0，其解析梯度f'(x)=0;
%若h>0,且正好h>|x|,则f(x+h)=x+h>0（越过不可导点0）,其数值梯度f'(x+h)=1>0,与解析梯度不一致（f(x-h)同理）
%但是，在不可导点附近的x只是少数，如果sigmoid或者tanh激活函数的梯度检验通过，且relu激活函数的大部分数值梯度检验通过（有时会全部通过），则梯度计算没有问题

%训练参数定义
opts.alpha = 0.01;   %学习率
opts.momentum = 0.9;  %动量项权值
opts.batchnum = 4; %批大小
opts.numepochs = 20;  %迭代次数

%训练数据
data = rand(6,7,3);
label = eye(4,3);
inputSize = size(data);
outputSize = 4;

%运行
cnn= nn_setup(cnn, inputSize, outputSize);
cnn = nn_forward(cnn,data);
cnn = nn_backward(cnn,label);
%nn_grad_check(cnn,data,label);
%一定要跟新一次权重才能确保梯度算法运行正确，避免偶然性
cnn = nn_weight_update(cnn, opts);
cnn = nn_forward(cnn,data);
cnn = nn_backward(cnn,label);
nn_grad_check(cnn,data,label);
