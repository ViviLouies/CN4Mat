function bnlayer = bn_forward(inputMap,bnlayer,phase)
%inputMap 输入图cell格式：cell(inputmaps), e.g. size(cell(1)) = [height*width*datanum]
%bnlayer batch normalization层(参数)
%phase  'train' or 'test'
%注：
%bnlayer.z = z_score(prelayer.a)
%bnlayer.a = bnlayer.gamma * bnlayer.z + bnlayer.beta

switch phase
    case 'train' %除了前向传播外，还要记录每一个batch的均值和方差
        if bnlayer.flag %表示BN层在全连接层中
            bnlayer.mean = mean(inputMap,2); %特征图的均值(按行计算,列向量形式)
            bnlayer.z_decent = bsxfun(@minus,inputMap,bnlayer.mean); %特征图的每一行去中心化
            bnlayer.var = mean((bnlayer.z_decent).^2,2);  %特征图的方差(有偏估计,列向量形式)
            bnlayer.std = sqrt(bnlayer.var + bnlayer.epsilion); %特征图的标准差（有偏，epsilion平滑）
            bnlayer.z_norm = bsxfun(@rdivide,bnlayer.z_decent,bnlayer.std); % z-score标准化
            %MATLAB函数：[z_norm,mean,std] = zscore(inputMap,1,2);
            %映射重构
            bnlayer.a = bsxfun(@times,bnlayer.z_norm,bnlayer.gamma);
            bnlayer.a = bsxfun(@plus,bnlayer.a,bnlayer.beta);
            %记录每一个batch的均值和方差
            if isempty(bnlayer.all_mean)  %第一次
                bnlayer.all_mean(:,1) = bnlayer.mean;
                bnlayer.all_var(:,1) = bnlayer.var;
            else
                bnlayer.all_mean(:,end+1) = bnlayer.mean;
                bnlayer.all_var(:,end+1) = bnlayer.var;
            end
        else %表示BN层在卷积层中
            for i = 1:bnlayer.featuremaps
                bnlayer.mean{i,1} = mean(inputMap{i,1},3); %特征图的均值(矩阵形式)
                bnlayer.z_decent{i,1} = bsxfun(@minus,inputMap{i,1},bnlayer.mean{i,1}); %特征图的每个slice去中心化
                bnlayer.var{i,1} = mean((bnlayer.z_decent{i,1}).^2,3);  %特征图的方差(有偏估计,矩阵形式)
                bnlayer.std{i,1} = sqrt(bnlayer.var{i,1}+bnlayer.epsilion); %特征图的标准差（有偏，epsilion平滑）
                bnlayer.z_norm{i,1} = bsxfun(@rdivide, bnlayer.z_decent{i,1}, bnlayer.std{i,1}); % z-score标准化
                %映射重构
                bnlayer.a{i,1} = bsxfun(@times, bnlayer.z_norm{i,1}, bnlayer.gamma(i,1));
                bnlayer.a{i,1} =  bsxfun(@plus, bnlayer.a{i,1}, bnlayer.beta(i,1)); 
                %记录每一个batch的均值和方差
                if isempty(bnlayer.all_mean{i,1})  %第一次
                    bnlayer.all_mean{i,1}(:,:,1) = bnlayer.mean{i,1};
                    bnlayer.all_var{i,1}(:,:,1) = bnlayer.var{i,1};
                else
                    bnlayer.all_mean{i,1}(:,:,end+1) = bnlayer.mean{i,1};
                    bnlayer.all_var{i,1}(:,:,end+1) = bnlayer.var{i,1};
                end
            end
        end
    case 'test'  %计算所有训练数据的均值和方差
        if bnlayer.flag %表示BN层在全连接层中
            batchnum = size(inputMap,2);
            bnlayer.mean = mean(bnlayer.all_mean,2); %计算所有训练数据的均值(按行计算,列向量形式)
            bnlayer.z_decent = bsxfun(@minus,inputMap,bnlayer.mean); %特征图的每一行去中心化
            bnlayer.var = batchnum ./(batchnum-1) .* mean(bnlayer.all_var,2);  %计算所有训练数据的方差（无偏估计，列向量形式）
            bnlayer.std = sqrt(bnlayer.var + bnlayer.epsilion); %特征图的标准差（无偏，epsilion平滑）
            bnlayer.z_norm = bsxfun(@rdivide,bnlayer.z_decent,bnlayer.std); % z-score标准化
            bnlayer.a = bsxfun(@times,bnlayer.z_norm,bnlayer.gamma);  %映射重构
            bnlayer.a = bsxfun(@plus,bnlayer.a,bnlayer.beta);
        else %表示BN层在卷积层中
            batchnum = size(inputMap{1,1},3);
            for i = 1:bnlayer.featuremaps
                bnlayer.mean{i,1} = mean(bnlayer.all_mean{i,1},3); %%计算所有训练数据的均值(矩阵形式)
                bnlayer.z_decent{i,1} = bsxfun(@minus,inputMap{i,1},bnlayer.mean{i,1}); %特征图的每个slice去中心化
                bnlayer.var{i,1} = batchnum ./(batchnum-1) .* mean(bnlayer.all_var{i,1},3);  %计算所有训练数据的方差(无偏估计,矩阵形式)
                bnlayer.std{i,1} = sqrt(bnlayer.var{i,1}+bnlayer.epsilion); %特征图的标准差（无偏，epsilion平滑）
                bnlayer.z_norm{i,1} = bsxfun(@rdivide, bnlayer.z_decent{i,1}, bnlayer.std{i,1}); % z-score标准化
                bnlayer.a{i,1} = bsxfun(@times, bnlayer.z_norm{i,1}, bnlayer.gamma(i,1));
                bnlayer.a{i,1} =  bsxfun(@plus, bnlayer.a{i,1}, bnlayer.beta(i,1)); %映射重构
            end
        end
    otherwise
        error('Undefined phase of batch normalization layer: %s!',phase);
end
end