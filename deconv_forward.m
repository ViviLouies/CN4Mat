function outputMap = deconv_forward(inputMap,deconvlayer)
%inputMap 输入图cell格式：cell(inputmaps), e.g. size(cell(1)) = [height*width*datanum]
%deconvlayer 转置卷积层(参数)
%outputMap 转置卷积输出结果，还是cell格式
%注：
%deconvlayer.z = deconv(deconvlayer.w,prerlayer.a)
%deconvlayer.a = deconvlayer.z

inputmaps = numel(inputMap); %读入inputmaps数目
inputsize = size(inputMap{1});  %读入inputmaps大小
mapsize = inputsize(1:2);
batchnum = inputsize(3);
outputsize = deconvlayer.mapsize ;  %featuremaps的尺寸
outputMap = cell(deconvlayer.featuremaps,1);   %预先开辟featuremaps的存储空间
for i = 1 : deconvlayer.featuremaps   %卷积核的数目，即featuremaps数目
    convtemp = zeros(outputsize(1),outputsize(2),batchnum);  %临时变量，保存一个inputmap卷积后的结果featuremaps
    for j = 1:inputmaps %输入通道数
        padMap = map_padding(inputMap{j,1},mapsize,deconvlayer.kernelsize,deconvlayer.pad,deconvlayer.stride);
        %根据mapsize,pad和stride填充前一层的输入featuremaps
        convtemp = convtemp + convn(padMap,rot180(deconvlayer.w{i,j}), 'valid'); %一个卷积核依次卷积每一个inputmaps(步长为1)
        %convn函数会自动旋转卷积核,所以要预先旋转一下
    end
    outputMap{i,1} = convtemp + deconvlayer.b{i,1}; %加偏置（纯线性）
end
