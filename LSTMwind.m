clc; clear;
%% load the data 【数据加载】
load('traindata.mat');

data = traindata;
inputdata  = [data.LowPassFiltered, data.HighPassFiltered];
targetdata = data.OriginalData;     % 目标数据

trainLen = 725;       % 训练数据集长度
testLen  = 365;       % 测试数据集长度
initLen  = 0;         % 预热长度
predictInterval = 1;  % 预测间隔

% 时间序列
traintime = data.Date(2 + initLen : trainLen);                % 训练集时间
testtime  = data.Date(trainLen + 2 : trainLen + 1 + testLen); % 测试集时间

% 训练数据 输入[1:725] 输出[1+1:725+1] 输入[1:725] 输出[1+5:725+5]
Xtrain = inputdata(1 : trainLen, :)';                          
Ytrain = targetdata(1 + predictInterval : trainLen + predictInterval)'; 

% 测试数据 输入[725+1: 725 + 365] 输出[725+1+1: 725 + 365 + 1]
Xtest = inputdata(trainLen + 1 : trainLen + testLen, :)';
Ytest = targetdata(trainLen + 1 + predictInterval : trainLen + testLen + predictInterval)';

%% 定义LSTM网络结构
numFeatures = 2;
numResponses = 1;
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

% 训练选项
options = trainingOptions('adam', ...
    'MaxEpochs',1000, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.95, ...
    'Verbose',0, ...
    'Plots','training-progress');

% 训练模型
net = trainNetwork(Xtrain,Ytrain,layers,options);

% 在训练集上进行预测
net = predictAndUpdateState(net,Xtrain);
YPredTrain = predict(net,Xtrain,'ExecutionEnvironment','gpu');

% 计算训练误差
mseTrain = mean((YPredTrain-Ytrain).^2);

% 在测试集上进行预测
net = resetState(net);

%% 使用实际的测试数据进行预测
numTimeStepsTest = size(Xtest, 2);
YPredTest = zeros(numTimeStepsTest, 1);

for i = 1:numTimeStepsTest
    [net, YPredTest(i)] = predictAndUpdateState(net, Xtest(:, i),'ExecutionEnvironment','gpu');
end

%% 使用预测的数据预测下一步
% net = predictAndUpdateState(net,XTrain);
% [net,YPredTest] = predictAndUpdateState(net,YTrain(end));
% 
% numTimeStepsTest = numel(XTest);
% for i = 2:numTimeStepsTest
%     [net,YPredTest(:,i)] = predictAndUpdateState(net,YPredTest(:,i-1),'ExecutionEnvironment','gpu');
% end


% 计算测试误差
mseTest = sum(mean((YPredTest-Ytest).^2));

% 绘制训练误差和测试误差对比图
figure
subplot(2,1,1);
plot(Ytrain, 'r');
hold on;
plot(YPredTrain, 'b');
hold off;
legend(["Actual" "Predicted"]);
xlabel("Time Step");
ylabel("Wind Speed");
title("LSTM Training Data: Actual vs Predicted");

subplot(2,1,2);
plot(Ytest, 'r');
hold on;
plot(YPredTest, 'b');
hold off;
legend(["Actual" "Predicted"]);
xlabel("Time Step");
ylabel("Wind Speed");
title("LSTM Test Data: Actual")
