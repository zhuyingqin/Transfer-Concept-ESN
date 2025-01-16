% 遗传算法优化concept，并且丢弃一定比例训练数据
clc;
clear;
close all;

%% load the data 【数据加载】
load('traindata.mat');

data = traindata;
% inputdata  = [data.LowPassFiltered, data.HighPassFiltered];
% targetdata = data.OriginalData;     % 目标数据

[seasonal, trend] = seriesDecomp(data.OriginalData, 5);

inputdata  = [seasonal, trend];
targetdata = data.OriginalData;     % 目标数据

deletePercentage = 0.1;    % 要删除的数据比例
numToDelete = round(725 * deletePercentage); % 计算要删除的行数
trainLen = 725;             % 训练数据集长度
testLen  = 365;             % 测试数据集长度
initLen  = 0;               % 预热长度
predictInterval = 1;        % 预测间隔

% 时间序列
traintime = data.Date(2 + initLen : trainLen);                % 训练集时间
testtime  = data.Date(trainLen + 2 : trainLen + 1 + testLen); % 测试集时间

% 训练数据 输入[1:725] 输出[1+1:725+1] 输入[1:725] 输出[1+5:725+5]
Xtrain = inputdata(1 : trainLen, :);                          
Ytrain = targetdata(1 + predictInterval : trainLen + predictInterval); 
indicesToDelete = randperm(size(Xtrain, 1), numToDelete); % 随机选择要删除的行索引
Xtrain(indicesToDelete, :) = [];    % 删除这些行
Ytrain(indicesToDelete) = [];       % 同样也要从Ytrain中删除对应的行
trainLen = 725 - numToDelete;

% 测试数据 输入[725+1: 725 + 365] 输出[725+1+1: 725 + 365 + 1]
Xtest = inputdata(trainLen + 1 : trainLen + testLen, :);
Ytest = targetdata(trainLen + 1 + predictInterval : trainLen + testLen + predictInterval);


%% generate the ESN reservoir              【生成ESN的储藏层】
inSize  = size(inputdata,  2); % 输入节点数量，根据输入数据列数确定
outSize = size(targetdata, 2); % 输出节点数量，根据目标数据列数确定

resSize = 100;                 % 池内节点数

matrixType = 'sparse';         % 可以设置为 'dense' 或者 'sparse'
switch matrixType
    case 'dense'
        % 密集的W
        W = rand(resSize, resSize) - 0.5; % 池内权值
    case 'sparse'
        % 稀疏的W
        density = 0.2; % 设置稀疏矩阵的密度
        W = sprand(resSize, resSize, density); % 生成稀疏矩阵并调整值的范围
    otherwise
        error('未知的矩阵类型。请选择 "dense" 或 "sparse".');
end

%% 遗传算法

% 遗传算法参数
numWin = resSize * (1 + inSize);
numC   = resSize * resSize;
numWb  = resSize;

numVariables = numWin + numC + numWb + 2; % Win, W, Wb, a, C
lb = -1 * ones(1, numVariables); % 参数的下界
ub =  1 * ones(1, numVariables); % 参数的上界
options = optimoptions('ga', 'PopulationSize', 50, 'MaxGenerations', 10, ...
    'Display', 'iter', 'UseParallel', true);

% 运行遗传算法(根据IF语句选择是否执行遗传算法，不执行就直接调用数据)
optimal = "GA";

if optimal == "GA"
    [optimalParams, optimalError] = ga(@(params) objectiveFunction(params, ...
        Xtrain, Ytrain, trainLen, initLen, inSize, resSize, W), ...
        numVariables, [], [], [], [], lb, ub, [], options);
    % 保存参数
    fileName = sprintf('optimalESNParams_Interval_%d.mat', predictInterval);
    save("Parameters/"+fileName);
else
    fileName = sprintf('optimalESNParams_Interval_%d.mat', predictInterval);
    load("Parameters/"+fileName);
end
[Win, C, Wb, a, spectralRadius] = decodeParams(optimalParams, inSize, resSize);

%% normalizing and setting spectral radius
disp 'Computing spectral radius...';         % 计算谱半径
opt.disp = 0;                                % 
rhoW = abs(eigs(W,1,'LM',opt));              % 谱半径W的最大特征值的绝对值
disp 'done.'                                 % 结束
W = W .* ( spectralRadius / rhoW);           % 

% allocated memory for the design (collected states) matrix 矩阵内存分配
X = zeros(resSize+1, trainLen-initLen);              %  
X_origin = zeros(resSize+1, trainLen-initLen);       %
%% run the reservoir with the data and collect X
%  更新池并收集X
x = zeros(resSize, 1);                       % 初始化储藏层矩阵
x_origin = zeros(resSize, 1);
Yb = 0;
for t = 1: trainLen
	u = Xtrain(t, :);                        % 输入值
    x_origin = (1-a)*x  + a*tanh( Win*[1 u]' + W*x + Wb' * Yb);
	x = C*x_origin;
    if t > initLen                           % 大于初始化时
		X(:,t-initLen) = [1 x'];             % 收集初始化后的[1; x; u]
        X_origin(:,t-initLen) = [1 x_origin'];
    end
    Yb = Ytrain(t);
end

%% train the output by ridge regression【通过岭回归训练输出】
reg = 1e-4;
Wout = ((X*X' + reg*eye(resSize+1)) \ (X*Ytrain));  % 公式27.9

train = Wout' * X;

Y = zeros(outSize, testLen-predictInterval);
for t = 1:testLen
    u = Xtest(t, :);
    x = (1-a)*x + a*tanh( Win*[1 u]' + W*x + Wb'*Yb);
    x = C*x;
    y = Wout'*[1 x']';                     
    if y <= 0
       y = 0; 
    end
    Y(:,t) = y;
    Yb = Ytest(t);
end

mseTest  = nrmse(Ytest', Y)';
mseTrain = nrmse(train, Ytrain');

% 绘制训练误差和测试误差对比图
% figure(1);
% plot(X');
% hold on;
% plot(X_origin');

% 绘制训练误差和测试误差对比图
fig = figure(2);
subplot(5,1,1);
plot(Ytrain, 'r');
hold on;
plot(train, 'b');
text(10, max(train), ['MSE: ' num2str(mseTrain)]); % 显示MSE
hold off;
legend(["Actual" "Predicted"]);
xlabel("Time Step");
ylabel("Wind Speed");
title(['ESN Training Data (Predict Interval: ' num2str(predictInterval) '): Actual vs Predicted']);

subplot(5,1,2);
hold on;
plot(Ytest, 'r');
plot(Y', 'b');
text(10, max(Ytest), ['MSE: ' num2str(mseTest)]); % 显示MSE
hold off;
legend(["Actual" "Predicted"]);
xlabel("Time Step");
ylabel("Wind Speed");
title(['ESN Test Data (Predict Interval: ' num2str(predictInterval) '): Actual vs Predicted']);


subplot(5,1,3);
plot(Ytest - Y', 'r');

xlabel("Time Step");
ylabel("Prediction error");
title(['Prediction error (Predict Interval: ' num2str(predictInterval) ')']);

subplot(5,1,4);
plot(train - Ytrain');

xlabel("Time Step");
ylabel("Training error");
title(['Training error (Predict Interval: ' num2str(predictInterval) ')']);

subplot(5,1,5)
plot(Xtest(1: end, 1))
hold on;
plot(Xtest(1: end, 2))
plot(Ytest)
legend('Lowpass', 'Highpass', 'Target')

% 定义文件名的基础部分
baseFilename = ['论文图/ESN_concept_' num2str(predictInterval)];

% 保存为PNG格式
pngFilename = [baseFilename '.png'];
saveas(fig, pngFilename);

% 保存为FIG格式（MATLAB图形）
figFilename = [baseFilename '.fig'];
saveas(fig, figFilename);

function error = objectiveFunction(params, Xtrain, Ytrain, trainLen, initLen, inSize, resSize, W)
    % 解析参数
    index = 0;
    Win = reshape(params(index + (1:resSize*(1+inSize))), [resSize, 1+inSize]);
    index = index + resSize*(1+inSize);
    C = reshape(params(index + (1:resSize^2)), [resSize, resSize]);
    index = index + resSize^2;
    Wb = params(index + (1:resSize));
    index = index + resSize;
    a = params(index + 1);
    spectralRadius = params(index + 2);
    
    % 调整水库矩阵 W 的谱半径
    opt.disp = 0;
    rhoW = max(abs(eigs(W, 1, 'LM', opt)));
    W = W * (spectralRadius / rhoW);

    % 初始化
    x = zeros(resSize, 1);                  % 初始化水库状态
    X = zeros(1+resSize, trainLen-initLen); % 初始化收集矩阵

    % 运行ESN
    Yb = 0;
    for t = 1:trainLen
        u = Xtrain(t, :); 
        x = (1-a)*x + a*tanh(Win*[1; u'] + W*x + Wb'*Yb);
        x = C*x;
        if t > initLen
            X(:, t-initLen) = [1; x];
        end
        Yb = Ytrain(t);
    end

    % 训练输出
    reg = 1e-3; % 正则化系数
    Wout = (X*X' + reg*eye(1+resSize)) \ (X*Ytrain(initLen+1:end));

    % 计算误差
    Ypred = Wout' * X;
    % error = mean((Ypred - Ytrain(initLen+1:end)').^2); % 均方误差
    error = nrmse(Ypred, Ytrain(initLen+1:end)');
end

function [Win, C, Wb, a, spectralRadius] = decodeParams(optimalParams, inSize, resSize)
    % 解析参数
    numWin = resSize * (1 + inSize);
    numW = resSize * resSize;
    numWb = resSize;
    Win = reshape(optimalParams(1:numWin), [resSize, 1+inSize]);
    C = reshape(optimalParams(numWin+1:numWin+numW), [resSize, resSize]);
    Wb = optimalParams(numWin + numW + 1:numWin + numW + numWb);
    a = optimalParams(end - 1);
    spectralRadius = optimalParams(end); % 新增谱半径
end





