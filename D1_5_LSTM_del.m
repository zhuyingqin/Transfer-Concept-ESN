% 遗传算法优化concept，并且丢弃一定比例训练数据
clc;
clear;
close all;
% parpool(4)

%% load the data 【数据加载】
index = 1;

% Define the filenames in a cell array
filenames = {'traindata.mat', 'traindata2.mat', 'traindata3.mat', ...
    'traindata4.mat', 'traindata5.mat', 'traindata6.mat', 'traindata7.mat'};

% Ensure the index is within the valid range
if index >= 1 && index <= length(filenames)
    filename = filenames{index};
else
    error('Index out of range');
end

wp1 = load('traindata.mat');
wp2 = load(filename);

data1 = wp1.traindata;
data2 = wp2.traindata;

inputdata1  = [data1.LowPassFiltered, data1.HighPassFiltered];
targetdata1 = data1.OriginalData;     % 目标数据

inputdata2  = [data2.LowPassFiltered, data2.HighPassFiltered];
targetdata2 = data2.OriginalData;     % 目标数据

deletePercentage = 0.1;     % 要删除的数据比例
trainLen = 725;             % 训练数据集长度
testLen  = 365;             % 测试数据集长度
initLen  = 0;               % 预热长度
predictInterval  = 1;       % 预测间隔

[Xtrain_1, Ytrain_1, Xtest_1, Ytest_1] = processData(trainLen, testLen, ...
    predictInterval, inputdata1, targetdata1, deletePercentage, 1);

[Xtrain_2, Ytrain_2, Xtest_2, Ytest_2] = processData(trainLen, testLen, ...
    predictInterval, inputdata2, targetdata2, deletePercentage, 2);
% trainLen = size(Ytrain_1, 1);

% 假设您已经有了用于训练的数据：XTrain 和 YTrain
% XTrain 是输入数据，YTrain 是标签或目标输出数据
% 例如，XTrain 可能是一个时间序列的历史数据，YTrain 是下一个时间步长的数据

% 1. 定义 LSTM 网络架构
numFeatures = size(Xtrain_1,2); % 特征数量，根据您的数据调整
numResponses = size(Ytrain_1,2); % 响应变量的数量，根据您的数据调整
numHiddenUnits = 200; % 隐藏层单元数量，可以调整

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(numResponses)
    regressionLayer];

% 2. 指定训练选项
options = trainingOptions('adam', ...
    'MaxEpochs',1000, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0);

% 3. 训练 LSTM 网络
net = trainNetwork(Xtrain_1',Ytrain_1',layers,options);

% 4. 使用训练好的网络进行预测
% 假设 XTest 是您的测试数据
YPred = predict(net,Xtest_1');

% 5. 评估模型性能
% 比较 YPred 和实际的标签数据 YTest，计算准确度或其他性能指标
mseTest   = Mse(YPred(50:end), Ytest_1(50:end)')
Rv = calculateRSquared(YPred(50:end), Ytest_1(50:end)')


figure(1);
plot(Ytest_1(10:end), 'r');
hold on;
plot(YPred(10:end)', 'b');
hold off;
legend(["Actual" "Predicted"]);
xlabel("Time Step");
ylabel("Wind Speed");
title(['LSTM Test Data (Predict Interval: ' num2str(predictInterval) '): Actual vs Predicted']);


function [Xtrain, Ytrain, Xtest, Ytest] = processData(trainLen, testLen, predictInterval, inputdata, targetdata, deletePercentage,seedd)
    rng(seedd);
    % 训练数据 输入[1:725] 输出[1+10:725+10]
    Xtrain = inputdata(1 : trainLen, :);                          
    Ytrain = targetdata(1 + predictInterval : trainLen + predictInterval); 

    % 测试数据 输入[725+1: 725 + 365] 输出[725+1+10: 725 + 365 + 10]
    Xtest = inputdata(trainLen + 1 : trainLen + testLen, :);
    Ytest = targetdata(trainLen + 1 + predictInterval : trainLen + testLen + predictInterval);
    
    numToDelete = round(trainLen * deletePercentage);   % 计算要删除的行数
    indicesToDelete = randperm(size(Xtrain, 1), numToDelete); % 随机选择要删除的行索引
    Xtrain(indicesToDelete, :) = [];    % 删除这些行
    Ytrain(indicesToDelete) = [];       % 同样也要从Ytrain中删除对应的行
end

function rSquared = calculateRSquared(y, y_hat)
    % y is the vector of observed values
    % y_hat is the vector of predicted values
    
    % Calculate the residual sum of squares (SSres)
    SSres = sum((y - y_hat).^2);
    
    % Calculate the total sum of squares (SStot)
    SStot = sum((y - mean(y)).^2);
    
    % Calculate R^2
    rSquared = 1 - (SSres/SStot);
end


