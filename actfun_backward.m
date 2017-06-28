function actfunlayer = actfun_backward(actfunlayer,postlayer)
%actfunlayer 激活函数层(参数)
%postlayer 后一层，计算残差
switch postlayer.type
    case'fc' %如果是fc，要分两种情况讨论
        if actfunlayer.flag %表示该激活函数层夹在全连接层（mapsize==[1,1]），直接乘以激活函数的偏导数即可
            switch actfunlayer.function
                case 'sigmoid'
                    actfunlayer.delta = postlayer.delta .* actfunlayer.a .* (1 - actfunlayer.a); %乘以sigmoid的偏导数
                case 'tanh'
                    actfunlayer.delta = postlayer.delta .* (1 - (actfunlayer.a).^2); %乘以tanh的偏导数
                case 'relu'
                    actfunlayer.delta = postlayer.delta .* double(actfunlayer.a>0.0); %乘以relu的偏导数
                    %对于relu函数而言：a=relu(z),a和z有相同的符号，这里以代替z
            end
        else  %否则表示该激活函数层存在隐含的光栅层（mapsize~=[1,1]），先进行反矢量化，再乘以激活函数的偏导数
            [height, width, batchnum] = size(actfunlayer.a{1}); %取前一层特征map尺寸
            maparea = height * width;
            for i = 1 : actfunlayer.featuremaps  %当前层的特征图的个数
                z = reshape(postlayer.delta((i - 1) * maparea + 1: i * maparea, :), height, width, batchnum); %反矢量化
                switch actfunlayer.function
                    case 'sigmoid'
                        actfunlayer.delta{i,1} = z .* actfunlayer.a{i,1} .* (1 - actfunlayer.a{i,1}); %乘以sigmoid的偏导数
                    case 'tanh'
                        actfunlayer.delta{i,1} = z .* (1 - (actfunlayer.a{i,1}).^2); %乘以tanh的偏导数
                    case 'relu'
                        actfunlayer.delta{i,1} = z .* double(actfunlayer.a{i,1}>0.0); %乘以relu的偏导数
                        %对于relu函数而言：a=relu(z),a和z有相同的符号，这里以代替z
                end
            end
        end
    case 'bn' %如果是bn，则前后两层特征图数目相同（各自相连），但也要分两种情况讨论  
        if actfunlayer.flag   %表示该激活函数层夹在全连接层中（mapsize==[1,1]），先求zscore的残差，再乘以激活函数的偏导数即可
            batchnum = size(postlayer.a,2);
            znorm_delta = 1 ./ repmat(postlayer.std,[1,batchnum]) .* (postlayer.delta - repmat(mean(postlayer.delta,2),[1,batchnum])...
                - repmat(mean(postlayer.delta .* postlayer.z_norm,2),[1,batchnum]) .* postlayer.z_norm);
            switch actfunlayer.function
                case 'sigmoid'
                    actfunlayer.delta = znorm_delta .* actfunlayer.a .* (1 - actfunlayer.a); %乘以sigmoid的偏导数
                case 'tanh'
                    actfunlayer.delta = znorm_delta .* (1 - (actfunlayer.a).^2); %乘以tanh的偏导数
                case 'relu'
                    actfunlayer.delta = znorm_delta .* double(actfunlayer.a>0.0); %乘以relu的偏导数
                    %对于relu函数而言：a=relu(z),a和z有相同的符号，这里以代替z
            end
        else  %表示该激活函数层夹在卷积层中（mapsize~=[1,1]）
            batchnum = size(postlayer.a{1},3);
            for i = 1 : actfunlayer.featuremaps  %当前层的特征图的个数（前后两层特征图数目相同）
                znorm_delta = 1 ./ repmat(postlayer.std{i,1},[1,1,batchnum]) .* (postlayer.delta{i,1} - repmat(mean(postlayer.delta{i,1},3),[1,1,batchnum])...
                - repmat(mean(postlayer.delta{i,1} .* postlayer.z_norm{i,1},3),[1,1,batchnum]) .* postlayer.z_norm{i,1});
                switch actfunlayer.function
                    case 'sigmoid'
                        actfunlayer.delta{i,1} = znorm_delta .* actfunlayer.a{i,1} .* (1 - actfunlayer.a{i,1}); %乘以sigmoid的偏导数
                    case 'tanh'
                        actfunlayer.delta{i,1} = znorm_delta.* (1 - (actfunlayer.a{i,1}).^2); %乘以tanh的偏导数
                    case 'relu'
                        actfunlayer.delta{i,1} = znorm_delta .* double(actfunlayer.a{i,1}>0.0); %乘以relu的偏导数
                        %对于relu函数而言：a=relu(z),a和z有相同的符号，这里以代替z
                end
            end
        end
    case 'pool' %如果是池化层，则前后两层特征图数目相同（各自相连），需先进行上采样，再乘以激活函数的偏导数
        for i = 1 : actfunlayer.featuremaps  %当前层的特征图的个数
            z = expand(postlayer.delta{i,1}, [postlayer.stride(1),postlayer.stride(2),1]) .* postlayer.maxPos{i,1}; %上采样
            switch actfunlayer.function
                case 'sigmoid'
                    actfunlayer.delta{i,1} = z .* actfunlayer.a{i,1} .* (1 - actfunlayer.a{i,1}); %乘以sigmoid的偏导数
                case 'tanh'
                    actfunlayer.delta{i,1} = z .* (1 - (actfunlayer.a{i,1}).^2); %乘以tanh的偏导数
                case 'relu'
                    actfunlayer.delta{i,1} = z .* double(actfunlayer.a{i,1}>0.0); %乘以relu的偏导数
                    %对于relu函数而言：a=relu(z),a和z有相同的符号，这里以代替z
            end
        end
    case 'conv' %如果是卷积层，需先进行反卷积，再乘以激活函数的偏导数
        for i = 1 : actfunlayer.featuremaps  %当前层的特征图的个数
            z = zeros(size(actfunlayer.a{1}));
            for j = 1 : postlayer.featuremaps %后一层特征图的个数
                padMap = map_padding(postlayer.delta{j,1},postlayer.mapsize,postlayer.kernelsize,postlayer.pad,postlayer.stride); 
                %根据mapsize,pad和stride填充后一层的灵敏度矩阵
                z = z + convn(padMap,postlayer.w{j,i}, 'valid'); %反卷积求和得到当前层的残差
                %由于convn会自动旋转卷积核，故这里不再旋转
            end
            switch actfunlayer.function
                case 'sigmoid'
                    actfunlayer.delta{i,1} = z .* actfunlayer.a{i,1} .* (1 - actfunlayer.a{i,1}); %乘以sigmoid的偏导数
                case 'tanh'
                    actfunlayer.delta{i,1} = z .* (1 - (actfunlayer.a{i,1}).^2); %乘以tanh的偏导数
                case 'relu'
                    actfunlayer.delta{i,1} = z .* double(actfunlayer.a{i,1}>0.0); %乘以relu的偏导数
                    %对于relu函数而言：a=relu(z),a和z有相同的符号，这里以代替z
            end
        end
    case 'deconv'  %如果是转置卷积层，需先进行反卷积，再乘以激活函数的偏导数
         for i = 1 : actfunlayer.featuremaps  %当前层的特征图的个数
            z = zeros(size(actfunlayer.a{1}));  %临时变量
            for j = 1 : postlayer.featuremaps %后一层特征图的个数
                a = convn(padarray(postlayer.delta{j,1},[postlayer.pad,0]),postlayer.w{j,i},'valid'); %一个卷积核依次卷积后一层每一个残差delta(这里假设步长为1)
                z = z + a(1:postlayer.stride(1):end,1:postlayer.stride(2):end,:); %根据步长采样,并求和(权值共享策略)
                %由于convn会自动旋转卷积核，故这里不再旋转
            end
            switch actfunlayer.function
                case 'sigmoid'
                    actfunlayer.delta{i,1} = z .* actfunlayer.a{i,1} .* (1 - actfunlayer.a{i,1}); %乘以sigmoid的偏导数
                case 'tanh'
                    actfunlayer.delta{i,1} = z .* (1 - (actfunlayer.a{i,1}).^2); %乘以tanh的偏导数
                case 'relu'
                    actfunlayer.delta{i,1} = z .* double(actfunlayer.a{i,1}>0.0); %乘以relu的偏导数
                    %对于relu函数而言：a=relu(z),a和z有相同的符号，这里以代替z
            end
         end
end


