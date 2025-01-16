%% 初始化
clear,clc,close all;
%% load the data 【数据加载】
load('traindata.mat');
rng(42)

data = traindata;
inputdata  = [data.LowPassFiltered, data.HighPassFiltered];
targetdata = data.OriginalData;     % 目标数据

trainLen = 725;       % 训练数据集长度
testLen  = 365;       % 测试数据集长度
initLen  = 0;         % 预热长度
predictInterval = 1; % 预测间隔

% 时间序列
traintime = data.Date(2 + initLen : trainLen);                % 训练集时间
testtime  = data.Date(trainLen + 2 : trainLen + 1 + testLen); % 测试集时间

% 训练数据 输入[1:725] 输出[1+1:725+1] 输入[1:725] 输出[1+5:725+5]
Xtrain = inputdata(1 : trainLen, :)';                          
Ytrain = targetdata(1 + predictInterval : trainLen + predictInterval)'; 

% 测试数据 输入[725+1: 725 + 365] 输出[725+1+1: 725 + 365 + 1]
Xtest = inputdata(trainLen + 1 : trainLen + testLen, :)';
Ytest = targetdata(trainLen + 1 + predictInterval : trainLen + testLen + predictInterval)';

%% 构建bp神经网络
net = newff(Xtrain,Ytrain,100);%一个隐含层，20个节点
%网络参数
net.trainParam.epochs=10000;%训练次数
net.trainParam.lr=0.001;%学习率
net.trainParam.goal=0.00001%训练最小误差
net.trainFcn = 'traingd';

%% bp神经网络训练
net = train(net,Xtrain,Ytrain);
%保存模型
save filename.net

%% bp神经网络测试
% 计算训练误差
an1=sim(net,Xtrain)%用训练好的模型进行仿真
mseTrain = mean((an1-Ytrain).^2);

an2=sim(net,Xtest)%用训练好的模型进行仿真
% 计算测试误差
mseTest = sum(mean((an2-Ytest).^2));

%% 绘制训练误差和测试误差对比图
figure
subplot(2,1,1);
plot(Ytrain, 'r');
hold on;
plot(an1, 'b');
hold off;
legend(["Actual" "Predicted"]);
xlabel("Time Step");
ylabel("Wind Speed");
title("LSTM Training Data: Actual vs Predicted");

subplot(2,1,2);
plot(Ytest, 'r');
hold on;
plot(an2, 'b');
hold off;
legend(["Actual" "Predicted"]);
xlabel("Time Step");
ylabel("Wind Speed");
title("LSTM Test Data: Actual")
