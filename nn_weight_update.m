function net = nn_weight_update(net, opts)
%opts.alpha  学习率
%opts.momentum  动量项权值
%只有含权值和偏置的网络层才需要更新权值
for layer = 2 : numel(net.layers)
    switch net.layers{layer}.type
        case  'conv' %卷积层权值更新
            for j = 1 : net.layers{layer}.featuremaps
                for i = 1: net.layers{layer-1}.featuremaps
                    %带动量项的SGD权重更新公式
                    net.layers{layer}.mw{j,i} = opts.momentum * net.layers{layer}.mw{j,i} - opts.alpha * net.layers{layer}.dw{j,i}; %计算动量项
                    net.layers{layer}.w{j,i} = net.layers{layer}.w{j,i} + net.layers{layer}.mw{j,i}; %权值更新
                    %简单SGD权值更新的公式：W_new = W_old - alpha * de/dW（SGD误差对权值导数）
                    %net.layers{layer}.w{outputmap,inputmap} = net.layers{layer}.w{outputmap,inputmap} - opts.alpha * net.layers{layer}.dw{outputmap,inputmap};
                end
                %带动量项的SGD偏置更新公式
                net.layers{layer}.mb{j} = opts.momentum * net.layers{layer}.mb{j} - opts.alpha * net.layers{layer}.db{j}; %计算动量项
                net.layers{layer}.b{j} = net.layers{layer}.b{j} + net.layers{layer}.mb{j}; %偏置更新
                %简单SGD偏置更新的公式：b_new = b_old - alpha * de/db（SGD误差对权值导数）
                %net.layers{layer}.b{outputmap} = net.layers{layer}.b{outputmap} - opts.alpha * net.layers{layer}.db{outputmap};
            end
        case 'deconv' %转置卷积层权值更新
            for j = 1 : net.layers{layer}.featuremaps
                for i = 1: net.layers{layer-1}.featuremaps
                    %带动量项的SGD权重更新公式
                    net.layers{layer}.mw{j,i} = opts.momentum * net.layers{layer}.mw{j,i} - opts.alpha * net.layers{layer}.dw{j,i}; %计算动量项
                    net.layers{layer}.w{j,i} = net.layers{layer}.w{j,i} + net.layers{layer}.mw{j,i}; %权值更新
                    %简单SGD权值更新的公式：W_new = W_old - alpha * de/dW（SGD误差对权值导数）
                    %net.layers{layer}.w{outputmap,inputmap} = net.layers{layer}.w{outputmap,inputmap} - opts.alpha * net.layers{layer}.dw{outputmap,inputmap};
                end
                %带动量项的SGD偏置更新公式
                net.layers{layer}.mb{j} = opts.momentum * net.layers{layer}.mb{j} - opts.alpha * net.layers{layer}.db{j}; %计算动量项
                net.layers{layer}.b{j} = net.layers{layer}.b{j} + net.layers{layer}.mb{j}; %偏置更新
                %简单SGD偏置更新的公式：b_new = b_old - alpha * de/db（SGD误差对权值导数）
                %net.layers{layer}.b{outputmap} = net.layers{layer}.b{outputmap} - opts.alpha * net.layers{layer}.db{outputmap};
            end
        case  'pool'%池化层权值更新
            if net.layers{layer}.weight
                for j = 1 : net.layers{layer}.featuremaps
                    net.layers{layer}.mw{j} = opts.momentum * net.layers{layer}.mw{j} - opts.alpha * net.layers{layer}.dw{j};
                    net.layers{layer}.w{j} = net.layers{layer}.w{j} + net.layers{layer}.mw{j};
                    net.layers{layer}.mb{j} = opts.momentum * net.layers{layer}.mb{j} - opts.alpha * net.layers{layer}.db{j};
                    net.layers{layer}.b{j} = net.layers{layer}.b{j} + net.layers{layer}.mb{j};
                    %简单SGD偏置更新的公式
                    %net.layers{layer}.w{outputmap} = net.layers{layer}.w{outputmap} - opts.alpha * net.layers{layer}.dw{outputmap};
                    %net.layers{layer}.b{outputmap} = net.layers{layer}.b{outputmap} - opts.alpha * net.layers{layer}.db{outputmap};
                    
                end
            end
        case  'bn' %batch normalization层权值更新
            net.layers{layer}.mgamma = opts.momentum .* net.layers{layer}.mgamma - opts.alpha .* net.layers{layer}.dgamma;%计算动量项
            net.layers{layer}.gamma = net.layers{layer}.gamma + net.layers{layer}.mgamma; %gamma更新
            net.layers{layer}.mbeta = opts.momentum .* net.layers{layer}.mbeta - opts.alpha * net.layers{layer}.dbeta;%计算动量项
            net.layers{layer}.beta = net.layers{layer}.beta + net.layers{layer}.mbeta; %beta更新
        case 'fc' %全连接层权值更新
            net.layers{layer}.mw = opts.momentum * net.layers{layer}.mw - opts.alpha * net.layers{layer}.dw;
            net.layers{layer}.w = net.layers{layer}.w + net.layers{layer}.mw ;
            net.layers{layer}.mb = opts.momentum * net.layers{layer}.mb - opts.alpha * net.layers{layer}.db;
            net.layers{layer}.b = net.layers{layer}.b + net.layers{layer}.mb;
             %简单SGD偏置更新的公式
            %net.layers{layer}.w = net.layers{layer}.w - opts.alpha * net.layers{layer}.dw;
            %net.layers{layer}.b = net.layers{layer}.b - opts.alpha * net.layers{layer}.db;
    end
end
end