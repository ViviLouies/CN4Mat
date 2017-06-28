function net = nn_backward(net,label)
%%计算残差（灵敏度）
%注：除卷积层外，其它层计算的残差都是其之前一层的残差
layer_num = numel(net.layers); %网络层数 
for layer = layer_num : -1 : 2
    switch net.layers{layer}.type
        case 'loss' %损失层
            net.layers{layer} = loss_backward(net.layers{layer},label,net.layers{layer-1}.w); %加入loss层前一层全连接层的连接权值（计算权重衰减,解决参数冗余）
            net.loss = net.layers{layer}.loss;
        case 'fc'  %全连接层
            net.layers{layer} = fc_backward(net.layers{layer},net.layers{layer+1}); %当前层和后一层
        case 'actfun' %激活函数层
            net.layers{layer} = actfun_backward(net.layers{layer},net.layers{layer+1}); %当前层和后一层
        case 'bn'  %batch normalization层
            net.layers{layer} = bn_backward(net.layers{layer},net.layers{layer+1}); %当前层和后一层
        case 'pool' %池化层
            net.layers{layer} = pool_backward(net.layers{layer},net.layers{layer+1}); %当前层和后一层
        case 'conv' %卷积层
            net.layers{layer} = conv_backward(net.layers{layer},net.layers{layer+1}); %当前层和后一层
        case 'deconv' %转置卷积层
            net.layers{layer} = deconv_backward(net.layers{layer},net.layers{layer+1}); %当前层和后一层
    end
end
%%计算梯度
net = nn_calc_weight(net);
end