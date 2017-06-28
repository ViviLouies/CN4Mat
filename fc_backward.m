function fclayer = fc_backward(fclayer,postlayer)
%fclayer 全连接层(参数)
%postlayer 后一层，计算残差
%残差反向传播，计算当前全连接层的灵敏度
switch postlayer.type
    case 'fc'
        fclayer.delta = fclayer.w' * postlayer.delta;
    case 'actfun'
        fclayer.delta = fclayer.w' * postlayer.delta;
    case 'loss'
        fclayer.delta = fclayer.w' * postlayer.delta;
    case 'bn'  %表示BN层夹在全连接层中（mapsize==[1,1]）,前后两层特征图数目相同（各自相连）
        batchnum = size(postlayer.a,2);
        fclayer.znorm_delta = 1 ./ repmat(postlayer.std,[1,batchnum]) .* (postlayer.delta - repmat(mean(postlayer.delta,2),[1,batchnum])...
            - repmat(mean(postlayer.delta .* postlayer.z_norm,2),[1,batchnum]) .* postlayer.z_norm); %先求zscore的残差
        fclayer.delta = fclayer.w' * fclayer.znorm_delta;  %再乘以全连接层连接权值
end