function fclayer = fc_forward(prelayer,fclayer)
%prelayer 全连接层之前一层(参数)
%fclayer 全连接层(参数)
%注：
%fclayer.z = prerlayer.a
%fclayer.a = fclayer.w * fclayer.z + fclayer.b

fc_in = [];
if strcmp(prelayer.type, 'conv') || strcmp(prelayer.type, 'deconv') || strcmp(prelayer.type, 'pool') || ...
   (strcmp(prelayer.type, 'bn') && ~(prelayer.flag)) || (strcmp(prelayer.type, 'actfun') && ~(prelayer.flag))
%如果全连接层的前一层是卷积层/池化层/batch normalization层/激活函数层（mapsize大小不是[1,1]，需要单独处理：矢量化）
    for i = 1:numel(prelayer.a) %前一层的featuremaps数目，即全连接层的inputmaps数目
        [height,width,batchnum] = size(prelayer.a{i});
        fc_in = [fc_in; reshape(prelayer.a{i},height*width,batchnum)]; %将前一层的输出maps展成列向量
    end
else
    fc_in = prelayer.a; %如果全连接层的前一层是全连接层（mapsize大小是[1,1]），则不用将前一层的outputmap展成向量
    batchnum = size(prelayer.a,2);
end
fclayer.input = fc_in; %保存前一层的输出结果（列向量形式）
fclayer.a = fclayer.w * fc_in + repmat(fclayer.b,[1,batchnum]); %计算输出结果（纯线性）
end