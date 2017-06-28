function outputMap = actfun_forward(inputMap,actfunlayer)
%inputMap 输入图cell格式：cell(inputmaps), e.g. size(cell(1)) = [height*width*datanum]
%actfunlayer 激活函数层(参数)
%outputMap 经传递函数输出结果，还是cell格式
%注：
%actfunlayer.z = prerlayer.a
%actfunlayer.a = actfunlayer.function(actfunlayer.z)
if actfunlayer.flag %表示该激活函数层夹在全连接层中（mapsize==[1,1]），直接计算激活函数
    switch actfunlayer.function
            case 'sigmoid'
                outputMap = 1./(1+exp(-inputMap)); %计算sigmoid结果
            case 'tanh'
                outputMap = tanh(inputMap); %计算tanh结果
            case 'relu'
                outputMap = inputMap .* (inputMap>0.0); %计算relu结果
            otherwise
                error('Unknown function of actfun layer: %s!',actfunlayer.function);
    end
else %表示该激活函数层夹在卷积层中（mapsize~=[1,1]）
    outputMap = cell(actfunlayer.featuremaps,1);   %预先开辟存储空间
    for i = 1 : actfunlayer.featuremaps   %featuremaps数目
        switch actfunlayer.function
            case 'sigmoid'
                outputMap{i,1} = 1./(1+exp(-inputMap{i,1})); %计算sigmoid结果
            case 'tanh'
                outputMap{i,1} = tanh(inputMap{i,1}); %计算tanh结果
            case 'relu'
                outputMap{i,1} = inputMap{i,1} .* (inputMap{i,1}>0.0); %计算relu结果
            otherwise
                error('Unknown function of actfun layer: %s!',actfunlayer.function);
        end
    end
end
end