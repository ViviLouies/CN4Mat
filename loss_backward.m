function losslayer = loss_backward(losslayer,label,w)
%fclayer loss层(参数)
%label 标签
%w loss层前一层全连接层的连接权值（计算权重衰减,解决参数冗余）
lambda = 1e-4; %softmax层的权重衰减系数
losslayer.error = losslayer.a - label; %实际输出与期望输出之间的误差
batchnum = size(losslayer.a, 2);
switch losslayer.function
    case 'sigmoid'
         losslayer.loss =  1/2* sum(losslayer.error(:) .^ 2) / batchnum;  %代价函数，采用均方误差函数作为代价函数
         losslayer.delta = losslayer.error .* (losslayer.a .* (1 - losslayer.a)); %输出层残差sigmoid传递函数
    case 'tanh'
        losslayer.loss =  1/2* sum(losslayer.error(:) .^ 2) / batchnum;  %代价函数，采用均方误差函数作为代价函数
        losslayer.delta = losslayer.error .* (1 - (losslayer.a).^2); %输出层残差tanh传递函数
    case 'relu'
        losslayer.loss =  1/2* sum(losslayer.error(:) .^ 2) / batchnum;  %代价函数，采用均方误差函数作为代价函数
        losslayer.delta = losslayer.error .* double(losslayer.a>0.0); %输出层残差relu传递函数（对于relu函数而言：a=relu(z),a和z有相同的符号，这里以代替z）
    case 'softmax'
        losslayer.loss = -1/batchnum * label(:)' * log(losslayer.a(:)) + lambda/2 * sum(w(:) .^ 2);  %softmax损失函数，加入权重衰减处理参数冗余(记得求权值梯度时算上这一项)
        losslayer.delta = losslayer.error;  %softmax层的灵敏度
end
end