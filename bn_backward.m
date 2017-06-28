function bnlayer = bn_backward(bnlayer,postlayer)
%bnlayer batch normalization层(参数)
%postlayer 后一层，计算残差
switch postlayer.type
    case 'fc' %如果是fc，要分两种情况讨论
        if bnlayer.flag  %表示BN层夹在全连接层中（mapsize==[1,1]），直接乘以gamma即可
            batchnum = size(bnlayer.a,2);
            bnlayer.delta = repmat(bnlayer.gamma,[1,batchnum]) .* postlayer.delta; %残差
            bnlayer.dgamma = sum(postlayer.delta .* bnlayer.z_norm,2) ./ batchnum;  %gamma偏导数
            bnlayer.dbeta = sum(postlayer.delta,2) ./ batchnum;  %beta偏导数
        else  %否则表示BN层存在隐含的光栅层中（mapsize~=[1,1]），先进行反矢量化，再乘以gamma即可
            [height, width, batchnum] = size(bnlayer.a{1}); %取前一层特征map尺寸
            maparea = height * width;
            for i = 1 : bnlayer.featuremaps  %当前层的特征图的个数
                z = reshape(postlayer.delta((i - 1) * maparea + 1: i * maparea, :), height, width, batchnum); %反矢量化
                bnlayer.delta{i,1} =  bnlayer.gamma(i,1) .* z;   %残差
                bnlayer.dgamma(i,1) = sum(sum(sum(z .* bnlayer.z_norm{i,1}))) ./ batchnum;  %gamma偏导数
                bnlayer.dbeta(i,1) = sum(z(:)) ./ batchnum;  %beta偏导数
            end
        end
    case'actfun' %如果是actfun，也要分两种情况讨论
        if bnlayer.flag  %表示BN层夹在全连接层中（mapsize==[1,1]），直接乘以gamma即可
            batchnum = size(bnlayer.a,2);
            bnlayer.delta = repmat(bnlayer.gamma,[1,batchnum]) .* postlayer.delta; %残差
            bnlayer.dgamma = sum(postlayer.delta .* bnlayer.z_norm,2) ./ batchnum;  %gamma偏导数
            bnlayer.dbeta = sum(postlayer.delta,2) ./ batchnum;  %beta偏导数
        else  %否则BN层夹在卷积层中（mapsize~=[1,1]），前后两层特征图数目相同（各自相连），直接乘以gamma即可
            batchnum = size(bnlayer.a{1},3);
            for i = 1 : bnlayer.featuremaps  %当前层的特征图的个数
                bnlayer.delta{i,1} = bnlayer.gamma(i,1) .* postlayer.delta{i,1};    %残差
                bnlayer.dgamma(i,1) = sum(sum(sum(postlayer.delta{i,1} .* bnlayer.z_norm{i,1}))) ./ batchnum;  %gamma偏导数
                bnlayer.dbeta(i,1) = sum(postlayer.delta{i,1}(:)) ./ batchnum;  %beta偏导数
            end
        end
    case 'pool' %如果是池化层，则前后两层特征图数目相同（各自相连），需先进行上采样，再乘以gamma即可
        batchnum = size(bnlayer.a{1},3);
        for i = 1 : bnlayer.featuremaps  %当前层的特征图的个数
            z = expand(postlayer.delta{i,1}, [postlayer.stride(1),postlayer.stride(2),1]) .* postlayer.maxPos{i,1}; %上采样
            bnlayer.delta{i,1} = bnlayer.gamma(i,1) .* z;     %残差
            bnlayer.dgamma(i,1) = sum(sum(sum(z .* bnlayer.z_norm{i,1}))) ./ batchnum;  %gamma偏导数
            bnlayer.dbeta(i,1) = sum(z(:)) ./ batchnum;  %beta偏导数
        end
    case 'conv' %如果是卷积层，需先进行反卷积，再乘以gamma即可
        batchnum = size(bnlayer.a{1},3);
        for i = 1 : bnlayer.featuremaps  %当前层的特征图的个数
            z = zeros(size(bnlayer.a{1}));
            for j = 1 : postlayer.featuremaps %后一层特征图的个数
                padMap = map_padding(postlayer.delta{j,1},postlayer.mapsize,postlayer.kernelsize,postlayer.pad,postlayer.stride); %根据mapsize,pad和stride填充后一层的灵敏度矩阵
                z = z + convn(padMap,postlayer.w{j,i}, 'valid'); %反卷积求和得到当前层的残差
                %由于convn会自动旋转卷积核，故这里不再旋转
            end
            bnlayer.delta{i,1} = bnlayer.gamma(i,1) .* z ;     %残差
            bnlayer.dgamma(i,1) = sum(sum(sum(z .* bnlayer.z_norm{i,1}))) ./ batchnum;  %gamma偏导数
            bnlayer.dbeta(i,1) = sum(z(:)) ./ batchnum;  %beta偏导数
        end
    case 'deconv'  %如果是转置卷积层，需先进行反卷积，再乘以bn层偏导数
         batchnum = size(bnlayer.a{1},3);
        for i = 1 : bnlayer.featuremaps  %当前层的特征图的个数
            z = zeros(size(bnlayer.a{1}));  %临时变量
            for j = 1 : postlayer.featuremaps %后一层特征图的个数
                a = convn(padarray(postlayer.delta{j,1},[postlayer.pad,0]),postlayer.w{j,i},'valid'); %一个卷积核依次卷积后一层每一个残差delta(这里假设步长为1)
                z = z + a(1:postlayer.stride(1):end,1:postlayer.stride(2):end,:); %根据步长采样,并求和(权值共享策略)
                %由于convn会自动旋转卷积核，故这里不再旋转
            end
            bnlayer.delta{i,1} = bnlayer.gamma(i,1) .* z ;     %残差
            bnlayer.dgamma(i,1) = sum(sum(sum(z .* bnlayer.z_norm{i,1}))) ./ batchnum;  %gamma偏导数
            bnlayer.dbeta(i,1) = sum(z(:)) ./ batchnum;  %beta偏导数
        end
end
