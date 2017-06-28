function net = nn_calc_weight(net)
%%计算梯度 
%loss层和actfun层没有权值，故不需要计算
%bn层的梯度已在反向传播过程中计算好了，故也不用在此计算
lambda = 1e-4; %softmax层的权重衰减系数
layer_num = numel(net.layers); %网络层数 
batchnum= size(net.layers{1}.a{1},3);
for layer = layer_num : -1 : 2
    switch net.layers{layer}.type
        case 'conv'  %卷积层
            for j = 1:net.layers{layer}.featuremaps
                for i = 1:net.layers{layer-1}.featuremaps
                    padMap = map_padding(net.layers{layer}.delta{j},net.layers{layer}.mapsize,[1,1],[0,0],net.layers{layer}.stride);
                    %考虑卷积层的步长，要将卷积层的灵敏度根据卷积步长进行内部扩充（其实这一步就是在按步长进行上采样，用0填充；由于不用进行外部填充，所以将kernelsize和pad设置为[1,1]和[0,0]）
                    z = convn(padarray(net.layers{layer-1}.a{i},[net.layers{layer}.pad,0]),flipall(padMap), 'valid');  %先外部补零（此可以和上一步map_padding同步），再卷积
                    %convn会自动旋转卷积核,这里要反旋转回来,flipall函数将矩阵的每一维度(这里是三个维度)都翻转了180度
                    net.layers{layer}.dw{j,i} = z./batchnum;
                end
                net.layers{layer}.db{j,1} = sum(net.layers{layer}.delta{j}(:)) / batchnum;
            end
        case 'deconv'  %转置卷积层
            for j = 1:net.layers{layer}.featuremaps
                for i = 1:net.layers{layer-1}.featuremaps
                    padMap = map_padding(net.layers{layer-1}.a{i},net.layers{layer-1}.mapsize,[1,1],[0,0],net.layers{layer}.stride);
                    %考虑卷积层的步长，要将卷积层的灵敏度根据卷积步长进行内部扩充（其实这一步就是在按步长进行上采样，用0填充；由于不用进行外部填充，所以将kernelsize和pad设置为[1,1]和[0,0]）
                    %z = rot180(convn(padarray(net.layers{layer}.delta{j},[net.layers{layer}.pad,0]),flipall(padMap), 'valid')); %先外部补零（此可以和上一步map_padding同步），再卷积
                    %convn会自动旋转卷积核,这里要反旋转回来,flipall函数将矩阵的每一维度(这里是三个维度)都翻转了180度
                    z = convn(rot180(padarray(net.layers{layer}.delta{j},[net.layers{layer}.pad,0])),flip(padMap,3), 'valid');
                    %另一种实现方式,减少翻转次数
                    net.layers{layer}.dw{j,i} = z./batchnum;
                end
                net.layers{layer}.db{j,1} = sum(net.layers{layer}.delta{j}(:)) / batchnum;
            end
        case 'pool' %池化层
            if net.layers{layer}.weight  %如果池化层有权值
                for j = 1:net.layers{layer}.featuremaps
                     %由于除卷积层外，其它层计算的残差都是其之前一层的残差，所以用layer+1层的残差，但这里用了个小技巧，比较pool前后两层之间残差的关系
                    delta_no_weight = net.layers{layer}.delta{j,1} ./ net.layers{layer}.w{j,1}; %由反向传播计算得到的池化层没有权值时的灵敏度（即有权值时的灵敏度除以权值即可）
                    downsample_no_weight = (net.layers{layer}.a{j,1} - net.layers{layer}.b{j,1}) ./ net.layers{layer}.w{j,1}; %池化层没有权值时的下采样结果(看作input)：downsample = (有权值时的下采样结果：a-偏置：b)./权值：w
                    net.layers{layer}.dw{j,1} = sum(delta_no_weight(:) .* downsample_no_weight(:)) ./ batchnum; %权值梯度
                    net.layers{layer}.db{j,1} = sum(delta_no_weight(:)) ./ batchnum;  %偏置梯度
                end
            end
        case 'fc' %全连接层
            %由于除卷积层外，其它层计算的残差都是其之前一层的残差，所以用layer+1层的残差
            if strcmp(net.layers{layer+1}.type,'bn') 
                 net.layers{layer}.dw =  net.layers{layer}.znorm_delta * (net.layers{layer}.input)'/ batchnum;  %权值梯度 
                 net.layers{layer}.db = mean(net.layers{layer}.znorm_delta, 2);  %偏置梯度
            else
                net.layers{layer}.dw = net.layers{layer+1}.delta * (net.layers{layer}.input)' / batchnum; %权值梯度 
                net.layers{layer}.db = mean(net.layers{layer+1}.delta, 2);  %偏置梯度
            end
            if strcmp(net.layers{layer+1}.type,'loss') && strcmp(net.layers{layer+1}.function,'softmax') %,如果是softmax损失函数，需加上权值衰减项
                net.layers{layer}.dw = net.layers{layer}.dw + lambda * net.layers{layer}.w;
            end
    end
end