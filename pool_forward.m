function poollayer = pool_forward(inputMap,poollayer)
%inputMap 输入图cell格式：cell(inputmaps), e.g. size(cell(1)) = [height*width*datanum]
%poollayer 池化层参数
%注：
%poollayer.z = downsample(prerlayer.a)
%poollayer.a =  poollayer.z OR poollayer.a = poollayer.w * poollayer.z + poollayer.b(if weight)

[height,width,batchnum] = size(inputMap{1});  %读入inputmaps大小
stride = poollayer.stride;   %步长默认[2,2]
poollayer.a = cell(poollayer.featuremaps,1); %预先开辟空间，保存池化结果（下采样）
%poollayer.downsample = cell(poollayer.featuremaps,1); %%预先开辟空间，仅保存下采样结果（便于池化层有权值时权值的更新）
poollayer.maxPos = cell(poollayer.featuremaps,1); %预先开辟空间，最大值位置记录，用于误差反向传播
%采用poollayer.maxPos标记方便后面统一两种情况的灵敏度计算
if strcmp(poollayer.method, 'max') %如果是最大池化
    for i = 1:poollayer.featuremaps
        poollayer.a{i,1} = zeros(height/stride(1),width/stride(2),batchnum); %初始化池化矩阵为0
        poollayer.maxPos{i,1} = zeros(height,width,batchnum); %初始化最大值位置矩阵为0
        for row = 1:stride(1):height
            for col = 1:stride(2):width
                patch = inputMap{i}(row:row+stride(1)-1,col:col+stride(2)-1,:); %patchsize:stride*stride*batchnum 
                [val,ind] = max(reshape(patch,[stride(1)*stride(2),batchnum]));  % 找出最大值及其位置
                %poollayer.downsample{i,1}((row+stride-1)/stride,(col+stride-1)/stride,:) = val; %保存下采样结果
                poollayer.a{i,1}((row+stride(1)-1)/stride(1),(col+stride(2)-1)/stride(2),:) = val;  %保存下采样结果
                if poollayer.weight %若池化层有权值
                    poollayer.a{i,1}((row+stride(1)-1)/stride(1),(col+stride(2)-1)/stride(2),:) = poollayer.w{i} .* val + poollayer.b{i};  % 加权重和偏置,max pooling,无激活函数（纯线性）
                end
                ind_row = rem(ind,stride(1)); %找到最大值索引对应的行坐标(共stride(1)*stride(2)个位置)
                ind_row(ind_row==0) = stride(1); %stride的倍数取余后为0，应加回去
                ind_col = ceil(ind/stride(1)); %找到最大值索引对应的列坐标
                for j = 1:batchnum
                    poollayer.maxPos{i,1}(row + ind_row(j) - 1, col + ind_col(j) - 1, j) = 1; %推出最大值位置在原图中的相应位置，置为1
                end
            end
        end
    end
elseif strcmp(poollayer.method, 'mean') %如果是平均池化
    for i = 1:poollayer.featuremaps
        z = convn(inputMap{i}, ones(stride) / (stride(1)*stride(2)), 'valid');   %用kron卷积实现平均池化
        z = z(1 : stride(1) : end, 1 : stride(2) : end, :); %根据采样步长跳读取值
        %poollayer.downsample{i,1} = z; %保存下采样结果
        poollayer.a{i,1} = z;  %保存下采样结果
        if poollayer.weight %若池化层有权值
            poollayer.a{i,1} = poollayer.w{i} .* z + poollayer.b{i};  %加权重,无激活函数
        end
        poollayer.maxPos{i,1} = 1/(stride(1)*stride(2)) .* ones(height,width,batchnum); %平均池化每个元素的概率都是1/(poollayer.scale^2)
    end
else
    error('Undefined method of pool layer: %s!',poollayer.method);
end
end