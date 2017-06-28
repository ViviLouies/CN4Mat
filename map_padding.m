function padMap = map_padding(delta,mapSize,kernelSize,pad,stride)
%delta：要填充的map
%mapSize：delta的图形尺寸
%kernelSize：卷积核的尺寸
%stride：卷积的步长
%padMap：输出填充（0）后的map
padMap = delta; %将padMap初始化为delta（如果不进行填充的话）
pad_out = kernelSize - pad - 1; %delta外部填充尺寸
pad_in = stride - 1;  %delta内部填充尺寸（即内部每两个元素间隔大小）
mapsize = mapSize + (mapSize-1) .* pad_in;
%delta内部扩充后的尺寸 = delta原尺寸 + 原来元素的间隔（mapSize-1）* 元素之间填充0的数目pad_in
%padSize = pad_out*2 + mapsize;
%delta扩充后的尺寸 = 外部填充尺寸*2 + 内部填充尺寸
datanum = size(delta,3);
%% 先填充内部(如果需要)
if sum(pad_in)
    map  = zeros([mapsize, datanum]);%初始化内部填充后的矩阵
    for i = 1:mapSize(1)
        for j = 1:mapSize(2)
            map(i+(i-1)*pad_in(1), j+(j-1)*pad_in(2),:)= delta(i,j,:); %按内部填充尺寸间隔填充
        end
    end
    padMap = map;
end
%% 再填充外部(如果需要)
if sum(pad_out)
    padMap = padarray(padMap,[pad_out,0]);  %用padarray函数填充外部
end
end