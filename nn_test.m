function [acc, ind] = nn_test(net, data, label)
    %%前向计算
    net = nn_forward(net, data, 'test');
    outlayer = numel(net.layers);
    [~, a] = max(net.layers{outlayer}.a); %找到网络最大值对应的类别
    [~, y] = max(label); %找到数据的真实类别
    ind = find(y ~= a);
    err = numel(ind) / size(label, 2); %计算错误率
    acc = 1.0 - err;
    fprintf('Testing accuracy = %f%%.\n',acc*100);
end