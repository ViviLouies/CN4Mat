function outputMap = conv_forward(inputMap,convlayer)
%inputMap 输入图cell格式：cell(inputmaps), e.g. size(cell(1)) = [height*width*datanum]
%convlayer 卷积层(参数)
%outputMap 卷积输出结果，还是cell格式
%注：
%convlayer.z = conv(convlayer.w,prerlayer.a)
%convlayer.a = convlayer.z

inputmaps = numel(inputMap); %读入inputmaps数目
batchnum = size(inputMap{1},3);  %读入inputmaps大小
outputsize = convlayer.mapsize ;  %featuremaps的尺寸
outputMap = cell(convlayer.featuremaps,1);   %预先开辟featuremaps的存储空间
for i = 1 : convlayer.featuremaps   %卷积核的数目，即featuremaps数目
    convtemp = zeros(outputsize(1),outputsize(2),batchnum);  %临时变量，保存一个inputmap卷积后的结果featuremaps
    for j = 1:inputmaps
        z = convn(padarray(inputMap{j,1},[convlayer.pad,0]),rot180(convlayer.w{i,j}),'valid'); %一个卷积核依次卷积每一个inputmaps(这里假设步长为1)
        %卷积神经网络中的卷积其实是数学意义上的相关corr操作，两者区别主要在于是否翻转卷积核。
        %convn函数会自动旋转卷积核,所以要预先旋转一下
        convtemp = convtemp + z(1:convlayer.stride(1):end,1:convlayer.stride(2):end,:); %根据步长采样,并求和(权值共享策略)
    end
    outputMap{i,1} = convtemp +convlayer.b{i,1}; %加偏置（纯线性）
end