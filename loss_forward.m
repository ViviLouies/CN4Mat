function losslayer = loss_forward(prelayer,losslayer)
%prelayer loss层之前一层(一定是全连接fc层）
%losslayer loss层(参数)
%注：
%losslayer.z = prerlayer.a
%losslayer.a = losslayer.function(losslayer.z)

losslayer.input = prelayer.a; %保存前一层（倒数第二层）的输出结果（列向量形式）
switch losslayer.function
    case 'sigmoid'
        losslayer.a = 1./(1+exp(-losslayer.input)); %计算sigmoid结果
    case 'tanh'
        losslayer.a = tanh(losslayer.input); %计算tanh结果(build-in)
    case 'relu'
        losslayer.a = losslayer.input.*(losslayer.input>0.0); %计算relu结果(不常用)
    case  'softmax'
        M = bsxfun(@minus,losslayer.input,max(losslayer.input, [], 1)); %max(input, [], 1)求出各列的最大值，输出一个行向量
        M = exp(M);
        losslayer.a = bsxfun(@rdivide, M, sum(M));  %计算输出label
    otherwise
        error('Undefined type of loss layer: %s!',losslayer.function);
end
end