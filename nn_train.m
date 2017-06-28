function net = nn_train(net, data, label, opts)
% net: 网络结构
% data：训练数据
% label：训练数据对应标签
% opts：网络训练参数，包括：
% opts.batchnum 批大小
% opts.numepochs 迭代次数
% opts.alpha 学习率
% opts.momentum 动量项

datanum = size(data, 3);  %训练样本个数
disp(['num of data = ' num2str(datanum)]);
batch_itr = floor(datanum / opts.batchnum);  
interval = ceil(opts.numepochs/3) + 1;
inc = 1;
momentum = [0.9,0.95,0.99]; %动量项每迭代interval次数时更新一次
time = zeros(opts.numepochs,1);  %程序运行时间
cost = zeros(opts.numepochs*batch_itr,1); %显示用训练误差
loss = zeros(opts.numepochs*batch_itr,1); %真实记录训练误差
for epoch = 1 : opts.numepochs  %对于每次迭代
    disp(['>>>epoch ' num2str(epoch) '/' num2str(opts.numepochs) ':']);
    fprintf('learning rate = %f \n',opts.alpha);
    fprintf('momentum = %f \n',opts.momentum);
    tic;  % 计时
    if rem(epoch,interval)==0
        opts.momentum = momentum(inc); %每interval次迭代更新一次动量项
        inc= inc + 1;
    end
    if rem(epoch,10)==0
        opts.alpha = opts.alpha * 0.2; %每10次迭代更新一次学习速率
    end
    index = randperm(datanum);  %打乱样本
    for itr = 1 : batch_itr
        %依次取出每一次训练用的样本
        batch_x = data(:, :, index((itr - 1) * opts.batchnum + 1 : itr * opts.batchnum));
        batch_y = label(:,index((itr - 1) * opts.batchnum + 1 : itr * opts.batchnum));
        %前向计算
        net = nn_forward(net, batch_x, 'train');
        %误差反向传播
        net = nn_backward(net, batch_y);
        %网络权值更新
        net = nn_weight_update(net, opts);
        %代价函数值，依次累加
        loss((epoch-1)*batch_itr + itr) = net.loss; %真实记录每一次的损失
        cost((epoch-1)*batch_itr + itr) = 0.99*cost(end) + 0.01*net.loss; %显示用，让loss曲线更平滑
    end
    time(epoch,1) = toc;
    fprintf('cost = %f \n',cost(end));
    disp(['runing time：',num2str(toc),'s']);
end
plot(cost);title('loss function');
save('training_procedure.mat','loss','time','net','opts');
end