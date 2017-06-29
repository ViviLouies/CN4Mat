function net = nn_setup(net, inputSize, outputSize)
%net 网络定义
%inputSize 输入map的尺寸(height * width * batchnum);
%注：每个特征图featuremap = map * batchnum， 每个map大小：height * width
%outputSize 标签(one of c形式)的尺寸，即c，也就是输出层神经元的个数，分多少个类，自然就有多少个输出神经元
%shape format:[featuremaps,height,width,batchnum]

mapsize = inputSize(1:2); batchnum = inputSize(3);
for layer = 1 : numel(net.layers)   % 对每一层进行判断并操作
    switch net.layers{layer}.type
        case 'input' %输入层
            net.layers{layer}.featuremaps = 1;   %输入层的特征图就1个，即原始图像
            net.layers{layer}.mapsize = mapsize; %输入层的特征图的每个slice大小([height,width])
            fprintf('layer:%d-%s\n',layer,'input');
            fprintf('\tshape: [%d, %d, %d, %d]\n',1,inputSize);
        case 'conv' %卷积层
            if ~isfield(net.layers{layer},'stride')%如果未定义步长，默认为1
                net.layers{layer}.stride = [1,1];
            elseif size(net.layers{layer}.stride) == 1 %或者只定义了一维步长，则两维步长相等
                net.layers{layer}.stride = [net.layers{layer}.stride,net.layers{layer}.stride]; 
            end
            if ~isfield(net.layers{layer},'pad')%如果未定义外部填充数目，默认为0
                net.layers{layer}.pad = [0,0];
            elseif size(net.layers{layer}.pad) == 1 %或者只定义了一维填充数目，则两维填充数目相等
                net.layers{layer}.pad = [net.layers{layer}.pad,net.layers{layer}.pad]; 
            end
            if size(net.layers{layer}.kernelsize) == 1 %或者只定义了一维卷积尺寸，则卷积核宽和高相等
                net.layers{layer}.kernelsize = [net.layers{layer}.kernelsize,net.layers{layer}.kernelsize]; 
            end
            pre_judge = net.layers{layer-1}.mapsize + net.layers{layer}.pad - net.layers{layer}.kernelsize; 
            %若前一层特征图只能进行一次卷积（即特征图+pad和卷积核同尺寸），则其步长只能限定为1
            if sum(pre_judge) == 0 && (net.layers{layer}.stride(1) > 1 || net.layers{layer}.stride(2) > 1)
                warning('%d-%s -> %d-%s : the convolutional outputmap just has only one element => the stride should less than 1',layer-1,net.layers{layer-1}.type,layer,'conv');
                net.layers{layer}.stride = [1,1];  %步长强制置1
            end
            net.layers{layer}.mapsize = (net.layers{layer-1}.mapsize + 2 .* net.layers{layer}.pad - net.layers{layer}.kernelsize) ./ net.layers{layer}.stride + 1; 
            %更新卷积层特征图的每个slice大小([height,width])
            assert(all(floor(net.layers{layer}.mapsize)==net.layers{layer}.mapsize), ['Layer ' num2str(layer) ' mapsize must be an integer. Actual: ' num2str(net.layers{layer}.mapsize)]);
            %不能整除时，报错，需更换卷积核大小
            kernelarea = prod(net.layers{layer}.kernelsize); %卷积核的面积，prod计算数组的连乘积, eg. prod([1,2,3]) = 1*2*3 = 6;
            fan_out = net.layers{layer}.featuremaps * kernelarea;  %连接到后一层卷积核的权值W参数个数
            fan_in = net.layers{layer-1}.featuremaps * kernelarea;  %连接到前一层卷积核的权值W参数个数
            for i = 1 : net.layers{layer}.featuremaps  %对于卷积层的每一个outputmap(等于卷积核的个数)
                for j = 1: net.layers{layer-1}.featuremaps %对于卷积层每一个inputmaps(等于前一层的featuremaps)
                    net.layers{layer}.w{i,j} = (rand(net.layers{layer}.kernelsize) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out)); %初始化卷积层权值(Xavier方法)
                    net.layers{layer}.mw{i,j} = zeros(net.layers{layer}.kernelsize);  %权值更新的动量项（权值），初始化为0
                end
                net.layers{layer}.b{i,1} = 0;  %初始化卷积核偏置为零,每个特征图一个bias,并非每个卷积核一个bias
                net.layers{layer}.mb{i,1}=0;   %权值更新的动量项（偏置），初始化为0
            end
            fprintf('layer:%d-%s\n',layer,'conv');
            fprintf('\tfeaturemaps: %d\n',net.layers{layer}.featuremaps);
            fprintf('\tkernelsize: [%d, %d]\n',net.layers{layer}.kernelsize);
            fprintf('\tpad: [%d, %d]\n',net.layers{layer}.pad);
            fprintf('\tstride: [%d, %d]\n',net.layers{layer}.stride);
            fprintf('\tshape: [%d, %d, %d, %d]\n ',net.layers{layer}.featuremaps,net.layers{layer}.mapsize,batchnum);
        case 'deconv' %转置卷积层
             if ~isfield(net.layers{layer},'stride')%如果未定义步长，默认为1
                net.layers{layer}.stride = [1,1];
            elseif size(net.layers{layer}.stride) == 1 %或者只定义了一维步长，则两维步长相等
                net.layers{layer}.stride = [net.layers{layer}.stride,net.layers{layer}.stride]; 
            end
            if ~isfield(net.layers{layer},'pad')%如果未定义外部填充数目，默认为0
                net.layers{layer}.pad = [0,0];
            elseif size(net.layers{layer}.pad) == 1 %或者只定义了一维填充数目，则两维填充数目相等
                net.layers{layer}.pad = [net.layers{layer}.pad,net.layers{layer}.pad]; 
            end
            if size(net.layers{layer}.kernelsize) == 1 %或者只定义了一维卷积尺寸，则卷积核宽和高相等
                net.layers{layer}.kernelsize = [net.layers{layer}.kernelsize,net.layers{layer}.kernelsize]; 
            end
            net.layers{layer}.mapsize = (net.layers{layer-1}.mapsize - 1) .* net.layers{layer}.stride + net.layers{layer}.kernelsize - 2 .* net.layers{layer}.pad; 
            %更新卷积层特征图的每个slice大小([height,width])
            assert(all(floor(net.layers{layer}.mapsize)==net.layers{layer}.mapsize), ['Layer ' num2str(layer) ' mapsize must be an integer. Actual: ' num2str(net.layers{layer}.mapsize)]);
            %不能整除时，报错，需更换卷积核大小
            kernelarea = prod(net.layers{layer}.kernelsize); %卷积核的面积，prod计算数组的连乘积, eg. prod([1,2,3]) = 1*2*3 = 6;
            fan_out = net.layers{layer}.featuremaps * kernelarea;  %连接到后一层卷积核的权值W参数个数
            fan_in = net.layers{layer-1}.featuremaps * kernelarea;  %连接到前一层卷积核的权值W参数个数
            for i = 1 : net.layers{layer}.featuremaps  %对于转置卷积层的每一个outputmap(等于卷积核的个数)
                for j = 1: net.layers{layer-1}.featuremaps %对于转置卷积层每一个inputmaps(等于前一层的featuremaps)
                    net.layers{layer}.w{i,j} = (rand(net.layers{layer}.kernelsize) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out)); %初始化卷积层权值(Xavier方法)
                    net.layers{layer}.mw{i,j} = zeros(net.layers{layer}.kernelsize);  %权值更新的动量项（权值），初始化为0
                end
                net.layers{layer}.b{i,1} = 0;  %初始化卷积核偏置为零,每个特征图一个bias,并非每个卷积核一个bias
                net.layers{layer}.mb{i,1}=0;   %权值更新的动量项（偏置），初始化为0
            end
            fprintf('layer:%d-%s\n',layer,'deconv');
            fprintf('\tfeaturemaps: %d\n',net.layers{layer}.featuremaps);
            fprintf('\tkernelsize: [%d, %d]\n',net.layers{layer}.kernelsize);
            fprintf('\tpad: [%d, %d]\n',net.layers{layer}.pad);
            fprintf('\tstride: [%d, %d]\n',net.layers{layer}.stride);
            fprintf('\tshape: [%d, %d, %d, %d]\n ',net.layers{layer}.featuremaps,net.layers{layer}.mapsize,batchnum);
        case 'pool' %池化层
            if strcmp(net.layers{layer-1}.type,'pool')
                error('%s -> %s connection is not supported!','pool','pool');%池化层后面不支持再接一个池化层
            end
            if ~isfield(net.layers{layer},'kernelsize')%如果未定义池化尺度（按照全卷积网络思想，即卷积核大小），默认为2
                net.layers{layer}.kernelsize = [2,2];
            elseif size(net.layers{layer}.kernelsize) == 1 %如果只定义了一维尺度，则步长两维相等
                net.layers{layer}.kernelsize = [net.layers{layer}.kernelsize,net.layers{layer}.kernelsize]; 
            end
            if sum(net.layers{layer}.kernelsize(:)) <= 2
                error('Pooling Layer %d stride should greater than [1,1] for non-overlapping convoluton!',layer);%池化层后面不支持再接一个池化层
            end
            net.layers{layer}.stride = net.layers{layer}.kernelsize;  %卷积核尺寸即步长（non-overlapping）
            net.layers{layer}.featuremaps = net.layers{layer-1}.featuremaps;  %池化层的特征图个数featuremaps和前一层一致
            net.layers{layer}.mapsize = net.layers{layer-1}.mapsize ./ net.layers{layer}.stride;   %更新池化层的featuremaps的每个slice大小
            assert(all(floor(net.layers{layer}.mapsize)==net.layers{layer}.mapsize), ['Layer ' num2str(layer) ' mapsize must be integer. Actual: ' num2str(net.layers{layer}.mapsize)]);
            %不能整除时，报错，需更换步长
            %注：池化层可以有以下3种情况：(1)没有参数(2)有权值(默认包含偏置)
            for i = 1 : net.layers{layer}.featuremaps   %对于层内的每个特征图
                if net.layers{layer}.weight %若池化层有权值
                    net.layers{layer}.w{i,1} = 1;  %初始化池化层的权重为1
                    net.layers{layer}.mw{i,1} = 0; %权值更新的动量项（权值），初始化为0
                    net.layers{layer}.b{i,1} = 0;  %初始化池化层的偏置为零
                    net.layers{layer}.mb{i,1} = 0;  %权值更新的动量项（偏置），初始化为0
                end
            end
            fprintf('layer:%d-%s\n',layer,'pool');
            fprintf('\tfeaturemaps: %d\n',net.layers{layer}.featuremaps);
            fprintf('\tkernelsize: [%d, %d]\n',net.layers{layer}.kernelsize);
            fprintf('\tstride: [%d, %d]\n',net.layers{layer}.stride);
            fprintf('\tmethod: %s\n',net.layers{layer}.method);
            fprintf('\tweight: %d\n',net.layers{layer}.weight);
            fprintf('\tshape: [%d, %d, %d, %d]\n ',net.layers{layer}.featuremaps,net.layers{layer}.mapsize,batchnum);
        case 'bn' %batch normalization层
            if strcmp(net.layers{layer-1}.type,'bn')
                error('%s -> %s connection is not supported!','bn','bn');%bn层后面不支持再接一个bn层
            end
            net.layers{layer}.featuremaps = net.layers{layer-1}.featuremaps;  %BN层的特征图个数featuremaps和前一层一致
            net.layers{layer}.mapsize = net.layers{layer-1}.mapsize;   %特征图每个slice的大小也一致
            net.layers{layer}.gamma = ones(net.layers{layer}.featuremaps,1);   %初始化映射重构权值gamma为1
            net.layers{layer}.mgamma = zeros(net.layers{layer}.featuremaps,1);  %初始化映射重构权值项的动量项为0
            net.layers{layer}.beta = zeros(net.layers{layer}.featuremaps,1);    %初始化映射重构偏置beta为0
            net.layers{layer}.mbeta = zeros(net.layers{layer}.featuremaps,1);   %初始化映射重构权值的动量项为0
            net.layers{layer}.epsilion = 1e-10; %标准差平滑项
            %标志位flag用来标记即bn层在卷积层（0）中，还是在全连接层（1）中
            if sum(net.layers{layer}.mapsize) == 2
                %特例：bn层在卷积层中，但其mapsize也是[1,1]
                if strcmp(net.layers{layer-1}.type,'conv') 
                    net.layers{layer}.flag = 0;
                elseif strcmp(net.layers{layer-1}.type,'pool')
                    net.layers{layer}.flag = 0;
                elseif strcmp(net.layers{layer-1}.type,'actfun') && net.layers{layer-1}.flag == 0
                    net.layers{layer}.flag = 0;
                else
                    net.layers{layer}.flag = 1; %在全连接层中
                end
            else
                net.layers{layer}.flag = 0;
            end
            if net.layers{layer}.flag
                net.layers{layer}.all_mean = []; %记录每一个batch的均值
                net.layers{layer}.all_var = [];  %记录每一个batch的方差
            else
                 for i=1:net.layers{layer}.featuremaps
                      net.layers{layer}.all_mean{i,1} =[];  %记录每一个batch的均值
                      net.layers{layer}.all_var{i,1} =[];   %记录每一个batch的方差
                 end
            end
            fprintf('layer:%d-%s\n',layer,'bn');
            fprintf('\tflag: %d\n',net.layers{layer}.flag);
            fprintf('\tshape: [%d, %d, %d, %d]\n ',net.layers{layer}.featuremaps,net.layers{layer}.mapsize,batchnum);
        case 'actfun' %激活函数层
            if strcmp(net.layers{layer-1}.type,'actfun')
                error('%s -> %s connection is not supported!','actfun','actfun');%激活函数层后面不支持再接一个激活函数层
            end
            if ~isfield(net.layers{layer},'function')%如果未定义激活函数，默认为sigmoid
                net.layers{layer}.function = 'sigmoid';
            end
            net.layers{layer}.featuremaps = net.layers{layer-1}.featuremaps;  %激活函数层的特征图个数featuremaps和前一层一致
            net.layers{layer}.mapsize = net.layers{layer-1}.mapsize;   %特征图每个slice的大小也一致
            %标志位flag用来标记即actfun层在卷积层（0）中，还是在全连接层（1）中
            if sum(net.layers{layer}.mapsize) == 2
                %特例：actfun层在卷积层中，但其mapsize也是[1,1]
                if strcmp(net.layers{layer-1}.type,'conv') 
                    net.layers{layer}.flag = 0;
                elseif strcmp(net.layers{layer-1}.type,'pool')
                    net.layers{layer}.flag = 0;
                elseif strcmp(net.layers{layer-1}.type,'bn') && net.layers{layer-1}.flag == 0
                    net.layers{layer}.flag = 0;
                else
                    net.layers{layer}.flag = 1; %在全连接层中
                end
            else
                net.layers{layer}.flag = 0;
            end
            fprintf('layer:%d-%s\n',layer,'actfun');
            fprintf('\tflag: %d\n',net.layers{layer}.flag);
            fprintf('\tfunction: %s\n',net.layers{layer}.function);
            fprintf('\tshape: [%d, %d, %d, %d]\n ',net.layers{layer}.featuremaps,net.layers{layer}.mapsize,batchnum);
        case 'fc'  %全连接层
            fcnum = prod(net.layers{layer-1}.mapsize) * net.layers{layer-1}.featuremaps;
            %fcnum 是前面一层的神经元个数,这一层的上一层可能是卷积层、池化层或BN层，包含有net.layers{layer-1}.featuremaps个特征图,每个特征图的大小是net.layers{layer-1}.mapsize
            %所以，该层的神经元个数是 特征map数目 * 每个特征map的大小（高和宽->若全连接层的前一层是卷积层、池化层或BN层，则长和宽可能大于1，若是全连接层，则长和宽均为1）
            %此操作又称矢量化（隐含的光栅层）
            net.layers{layer}.mapsize = [1,1];  %全连接层每个神经元的尺寸均为1*1
            net.layers{layer}.w= (rand(net.layers{layer}.featuremaps, fcnum) - 0.5) * 2 * sqrt(6 / (net.layers{layer}.featuremaps + fcnum));   %初始化全连接层权值(Xavier方法)
            net.layers{layer}.mw = zeros(net.layers{layer}.featuremaps, fcnum);  %权值更新的动量项，初始化为0
            net.layers{layer}.b= zeros(net.layers{layer}.featuremaps, 1);  %初始化全连接层偏置为0
            net.layers{layer}.mb = zeros(net.layers{layer}.featuremaps, 1);  %偏置更新的动量项，初始化为0
            fprintf('layer:%d-%s\n',layer,'fc');
            fprintf('\tweight_size: [%d, %d]\n',size(net.layers{layer}.w));
            fprintf('\tshape: [%d, %d, %d, %d]\n',net.layers{layer}.featuremaps,net.layers{layer}.mapsize,batchnum);
        case 'loss'  %损失层,即最后一层,只有一个激活函数，前一层必须是全连接层
            if ~strcmp(net.layers{layer-1}.type,'fc')
                error('a fc layer with outputsize of %d is required before loss layer, please add it!',outputSize);
            end
            if net.layers{layer-1}.featuremaps ~= outputSize
                error('the fc layer before loss layer should have the ''featuremaps'' of ''outputSize'', please fix it!');
            end
            if ~isfield(net.layers{layer},'function')%如果未定义激活函数，默认为sigmoid
                net.layers{layer}.function = 'sigmoid';
            end
            net.layers{layer}.featuremaps = 1;  %输出层的特征图就1个，即label
            net.layers{layer}.mapsize = [outputSize,1]; %输出层的每个slice大小,[height = outputSize, width =1]
            fprintf('layer:%d-%s\n',layer,'loss');
            fprintf('\tshape: [%d, %d, %d, %d]\n ',net.layers{layer}.featuremaps,net.layers{layer}.mapsize,batchnum);
        otherwise
            error('undefined type of layer!');
    end
end