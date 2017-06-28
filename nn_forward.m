function net = nn_forward(net, data, phase)
%网络前向传播
if nargin == 2
    phase = 'train'; %默认当前状态：训练
end
for layer = 1 : numel(net.layers)   %对于每层
    switch net.layers{layer}.type
        case 'input'  %输入层
            net.layers{layer}.a{1} = data; %网络的第一层就是输入数据，包含了多个训练图像，但只有一个特征图
        case 'conv'  %卷积层
            net.layers{layer}.a = conv_forward(net.layers{layer-1}.a,net.layers{layer}); %这里不考虑局部连接情况
        case 'deconv'  %转置卷积层
            net.layers{layer}.a = deconv_forward(net.layers{layer-1}.a,net.layers{layer}); %这里不考虑局部连接情况
        case 'pool'  %池化层
            net.layers{layer} = pool_forward(net.layers{layer-1}.a,net.layers{layer});
        case 'bn' %batch normalization层
            net.layers{layer} = bn_forward(net.layers{layer-1}.a,net.layers{layer},phase); %phase: train/test
        case 'actfun'  %激活函数层
            net.layers{layer}.a = actfun_forward(net.layers{layer-1}.a,net.layers{layer});
        case 'fc'  %全连接层
            net.layers{layer} = fc_forward(net.layers{layer-1},net.layers{layer});
        case 'loss' %损失层,即最后一层
            net.layers{layer} = loss_forward(net.layers{layer-1},net.layers{layer});
    end
end
