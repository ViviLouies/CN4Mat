function deconvlayer = deconv_backward(deconvlayer,postlayer)
%convlayer 卷积层(参数)
%postlayer 后一层，计算残差
switch postlayer.type
    case 'actfun' %如果是actfun,则前后两层特征图数目相同（各自相连），直接传递
        deconvlayer.delta = postlayer.delta;
    case 'bn'  %如果是bn，则前后两层特征图数目相同（各自相连），先求zscore的残差,再直接传递
        batchnum = size(postlayer.a{1},3);
        for i = 1 : deconvlayer.featuremaps  %当前层的特征图的个数
            znorm_delta = 1 ./ repmat(postlayer.std{i,1},[1,1,batchnum]) .* (postlayer.delta{i,1} - repmat(mean(postlayer.delta{i,1},3),[1,1,batchnum])...
                - repmat(mean(postlayer.delta{i,1} .* postlayer.z_norm{i,1},3),[1,1,batchnum]) .* postlayer.z_norm{i,1});
            deconvlayer.delta{i,1} = znorm_delta;
        end
    case 'fc' %如果是fc，表示当前层存在隐含的光栅层，需先进行反矢量化，直接传递
        [height, width, batchnum] = size(deconvlayer.a{1}); %取前一层特征map尺寸
        maparea = height * width;
        for i = 1 : deconvlayer.featuremaps  %当前层的特征图的个数
            deconvlayer.delta{i,1} = reshape(postlayer.delta((i - 1) * maparea + 1: i * maparea, :), height, width, batchnum); %反矢量化
        end
    case 'pool' %如果是pool，则前后两层特征图数目相同（各自相连），先上采样，再直接传递
        for i = 1 : deconvlayer.featuremaps  %当前层的特征图的个数
            deconvlayer.delta{i,1} = expand(postlayer.delta{i,1}, [postlayer.stride(1),postlayer.stride(2),1]) .* postlayer.maxPos{i,1}; %上采样
        end
    case 'conv'  %如果是卷积层，需先进行反卷积，再直接传递
        for i = 1 : deconvlayer.featuremaps  %当前层的特征图的个数
            z = zeros(size(deconvlayer.a{1}));
            for j = 1 : postlayer.featuremaps %后一层特征图的个数
                padMap = map_padding(postlayer.delta{j,1},postlayer.mapsize,postlayer.kernelsize,postlayer.pad,postlayer.stride); 
                %根据mapsize,pad和stride填充后一层的灵敏度矩阵
                z = z + convn(padMap,postlayer.w{j,i}, 'valid'); %反卷积求和得到当前层的残差
                %由于convn会自动旋转卷积核，故这里不再旋转
            end
            deconvlayer.delta{i,1} = z;  %直接传递
        end
   case 'deconv'  %如果是转置卷积层，则直接反卷积
         for i = 1 : deconvlayer.featuremaps  %当前层的特征图的个数
            z = zeros(size(deconvlayer.a{1}));  %临时变量
            for j = 1 : postlayer.featuremaps %后一层特征图的个数
                a = convn(padarray(postlayer.delta{j,1},[postlayer.pad,0]),postlayer.w{j,i},'valid'); %一个卷积核依次卷积后一层每一个残差delta(这里假设步长为1)
                z = z + a(1:postlayer.stride(1):end,1:postlayer.stride(2):end,:); %根据步长采样,并求和(权值共享策略)
                %由于convn会自动旋转卷积核，故这里不再旋转
            end
            deconvlayer.delta{i,1} = z;  %直接传递
        end
        
end