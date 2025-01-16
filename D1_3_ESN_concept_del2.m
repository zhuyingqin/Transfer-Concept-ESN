% 遗传算法优化concept，并且丢弃一定比例训练数据
clc;
clear;
close all;

%% load the data 【数据加载】
index = 4;

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

% [bestShift, Data2_aligned]   = findBestLagViaCrossCorrelation(inputdata1, inputdata2);
% [bestShift, target2_aligned] = findBestLagViaCrossCorrelation(targetdata1, targetdata2);
% inputdata2  = Data2_aligned;
% targetdata2 = target2_aligned;

deletePercentage = 0.1;     % 要删除的数据比例
trainLen = 725;             % 训练数据集长度
testLen  = 365;             % 测试数据集长度
initLen  = 0;               % 预热长度
predictInterval = 3;        % 预测间隔

[Xtrain_1, Ytrain_1, Xtest_1, Ytest_1] = processData(trainLen, testLen, ...
    predictInterval, inputdata1, targetdata1, deletePercentage);

[Xtrain_2, Ytrain_2, Xtest_2, Ytest_2] = processData(trainLen, testLen, ...
    predictInterval, inputdata2, targetdata2, deletePercentage);
trainLen = size(Ytrain_1, 1);

%% 生成ESN的储层
inSize  = size(inputdata1,  2); % 输入节点数量，根据输入数据列数确定
outSize = size(targetdata1, 2); % 输出节点数量，根据目标数据列数确定

resSize = 100;                   % 池内节点数

Win = (rand(resSize, 1+inSize) - 1) .* 1;    % 初始化输入
Wb  = rand(resSize, 1);
matrixType = 'sparse';          % 可以设置为 'dense' 或者 'sparse'
switch matrixType
    case 'dense'
        % 密集的 W
        W = rand(resSize, resSize) - 0.5;    % 池内权值
    case 'sparse'
        % 稀疏的 W
        density = 0.2; % 设置稀疏矩阵的密度
        W = sprand(resSize, resSize, density);
    otherwise
        error('未知的矩阵类型。请选择 "dense" 或 "sparse".');
end

% 构建ESN结构体[固定参数]
ESN1 = buildESNParameters(resSize, initLen, trainLen, testLen, matrixType, ...
        density, W, Win, Wb, Xtrain_1, Ytrain_1, Xtest_1, Ytest_1, inSize, outSize);

ESN2 = buildESNParameters(resSize, initLen, trainLen, testLen, matrixType, ...
        density, W, Win, Wb, Xtrain_2, Ytrain_2, Xtest_2, Ytest_2, inSize, outSize);

%% 遗传算法

% 运行遗传算法(根据IF语句选择是否执行遗传算法，不执行就直接调用数据)
optimal = "GA";

if optimal == "GA"
    numC   = resSize * resSize * 2;         % 
    numVariables = numC + 2;                % C1 C2 Row
    lb = -1 * ones(1, numVariables);        % 参数的下界
    ub =  1 * ones(1, numVariables);        % 参数的上界
    options = optimoptions('ga', 'PopulationSize', 75, 'MaxGenerations', 500, ...
    'Display', 'iter', 'UseParallel', true, 'FunctionTolerance', 1e-8, ...
    'MutationFcn', {@mutationuniform, 0.2});
    [optimalParams, optimalError1] = ga(@(params) objectiveFunction(params, ...
    ESN1, ESN2), numVariables, [], [], [], [], lb, ub, [], options);
    % 保存参数
    fileName = sprintf('ESNParams_Interval_del02_%d.mat', index);
    save("Parameters/"+fileName);

else
    fileName = sprintf('ESNParams_Interval_del02_%d.mat', index);
    load("Parameters/"+fileName);
    ESN1.a = 1;
    ESN2.a = 1 - ESN1.a;    
end

%% normalizing and setting spectral radius
[ESN1.C, ESN2.C, spectralRadius, a] = decodeParams(optimalParams, resSize);
ESN1.a = abs(a);
ESN2.a = 1-abs(a);

disp 'Computing spectral radius...';         % 计算谱半径
opt.disp = 0;                                % 
rhoW = abs(eigs(W,1,'LM',opt));              % 谱半径W的最大特征值的绝对值
disp 'done.'                                 % 结束
ESN1.Wres = W .* ( spectralRadius / rhoW);   % 
ESN2.Wres = W .* ( spectralRadius / rhoW);   % 

%% 通过岭回归训练输出
X = computeStateMatrix(ESN1, ESN2, "train");   % 计算训练的状态矩阵

reg = 1e-2;
Wout = ((X*X' + reg*eye(resSize)) \ (X*ESN1.Ytrain));    % 计算输出权重

train = Wout' * X;

X_test = computeStateMatrix(ESN1, ESN2, "test");
Y = Wout' * X_test;

mseTest   = Mse(Y(50:end), ESN1.Ytest(50:end)');
mseTrain  = Mse(train, ESN1.Ytrain');
rmseTest  = rmse(Y(50:end), ESN1.Ytest(50:end)')
rmseTrain = rmse(train, ESN1.Ytrain')
%% 绘制训练误差和测试误差对比图
fig = figure(2);
subplot(4,1,1);
plot(ESN1.Ytrain, 'r');
hold on;
plot(train, 'b');
text(10, max(train), ['MSE: ' num2str(mseTrain)]); % 显示MSE
hold off;
legend(["Actual" "Predicted"]);
xlabel("Time Step");
ylabel("Wind Speed");
title(['ESN Training Data (Predict Interval: ' num2str(predictInterval) '): Actual vs Predicted']);

subplot(4,1,2);
hold on;
plot(ESN1.Ytest(10:end), 'r');
plot(Y(10:end)', 'b');
text(10, max(ESN1.Ytest), ['MSE: ' num2str(mseTest)]); % 显示MSE
hold off;
legend(["Actual" "Predicted"]);
xlabel("Time Step");
ylabel("Wind Speed");
title(['ESN Test Data (Predict Interval: ' num2str(predictInterval) '): Actual vs Predicted']);


subplot(4,1,3);
plot(ESN1.Ytest - Y', 'r');

xlabel("Time Step");
ylabel("Prediction error");
title(['Prediction error (Predict Interval: ' num2str(predictInterval) ')']);

subplot(4,1,4);
plot(train - ESN1.Ytrain');

xlabel("Time Step");
ylabel("Training error");
title(['Training error (Predict Interval: ' num2str(predictInterval) ')']);

% 定义文件名的基础部分
baseFilename = ['论文图/ESN_concept_' num2str(predictInterval)];

% 保存为PNG格式
pngFilename = [baseFilename '.png'];
saveas(fig, pngFilename);

% 保存为FIG格式（MATLAB图形）
figFilename = [baseFilename '.fig'];
saveas(fig, figFilename);

function error = objectiveFunction(params, ESN1, ESN2)
    % 解析参数
    [C1, C2, spectralRadius, a] = decodeParams(params, ESN1.resSize);

    % 调整水库矩阵 W 的谱半径
    opt.disp = 0;
    rhoW = max(abs(eigs(ESN1.W, 1, 'LM', opt)));
    ESN1.Wres = ESN1.W * (spectralRadius / rhoW);
    ESN1.C    = C1;
    ESN1.a    = abs(a);

    ESN2.Wres = ESN2.W * (spectralRadius / rhoW);
    ESN2.C    = C2;
    ESN2.a    = 1-abs(a);
    
    X = computeStateMatrix(ESN1, ESN2, "train");
    
    % 训练输出
    reg = 1e-3; % 正则化系数
    Wout = (X*X' + reg*eye(ESN1.resSize)) \ (X*ESN1.Ytrain(ESN1.initLen+1:end));

    % 计算误差
    Ypred = Wout' * X;
    error = mean((Ypred - ESN1.Ytrain(ESN1.initLen+1:end)').^2); % 均方误差
    % error = nrmse(Ypred, Ytrain(initLen+1:end)');
end

function [C1, C2, spectralRadius, a] = decodeParams(params, resSize)
    % 解析参数
    index = 0;
    C1 = reshape(params(index + (1:resSize^2)), [resSize, resSize]);
    
    index = index + resSize^2;
    C2 = reshape(params(index + (1:resSize^2)), [resSize, resSize]);
    
    index = index + resSize^2;
    spectralRadius = params(index + 1);
    a = params(index + 2);
    % a = 1;
end

function ESN = buildESNParameters(resSize, initLen, trainLen, testLen, ...
    matrixType, density, W, Win, Wb, Xtrain, Ytrain, Xtest, Ytest, ...
    inSize, outSize)

    % 初始化ESN结构体
    ESN.inSize  = inSize;    % 输入尺寸
    ESN.outSize = outSize;   % 输出尺寸

    % 设置ESN参数
    ESN.resSize = resSize;              % 池内节点数
    ESN.initLen = initLen;              % 初始化长度
    ESN.trainLen = trainLen;            % 训练长度
    ESN.testLen = testLen;              % 测试长度
    ESN.matrixType = matrixType;        % 矩阵类型: 'dense' 或 'sparse'
    ESN.density = density;              % 稀疏矩阵的密度
    ESN.W       = W;
    ESN.Win     = Win;
    ESN.Wb      = Wb;
    ESN.Xtrain  = Xtrain;
    ESN.Ytrain  = Ytrain;
    ESN.Xtest   = Xtest;
    ESN.Ytest   = Ytest;
end

function X = computeStateMatrix(ESN, ESN2, type)
    % 从结构体中提取参数
    resSize  = ESN.resSize;
    initLen  = ESN.initLen;
    trainLen = ESN.trainLen;
    testLen  = ESN.testLen;
    Win      = ESN.Win;
    Wres     = ESN.Wres;
    Wb       = ESN.Wb;
    
    % 初始化储藏层状态矩阵
    x  = zeros(resSize, 1);
    x2 = zeros(resSize, 1);
    if type == "test"
        Len = testLen;
        X = zeros(resSize, testLen - initLen);
        Input    = ESN.Xtest;
        Output   = ESN.Ytest;
        Yb       = ESN.Ytrain(end);

        Input2   = ESN2.Xtest;
        Output2  = ESN2.Ytest;
        Yb2      = ESN2.Ytrain(end); 
    elseif type == "train"
        Len = trainLen;
        X = zeros(resSize, trainLen - initLen);
        Input    = ESN.Xtrain;
        Output   = ESN.Ytrain;
        Yb       = 0;

        Input2   = ESN2.Xtrain;
        Output2  = ESN2.Ytrain;
        Yb2      = 0;
    end

    % 运行储藏层并收集状态
    for t = 1:Len
        u  = Input(t, :); 
        u2 = Input2(t, :);
        
        x  = tanh(Win * [1 u]'  + Wres * x  + Wb*Yb);
        x2 = tanh(Win * [1 u2]' + Wres * x2 + Wb*Yb2);
        
        x  = ESN.a * ESN.C  * x;
        x2 = ESN2.a * ESN2.C * x2;
        if t > initLen
            X(:, t - initLen) = x + x2;
        end
        Yb  = Output(t);
        Yb2 = Output2(t);
    end

end

function [Xtrain, Ytrain, Xtest, Ytest] = processData(trainLen, testLen, predictInterval, inputdata, targetdata, deletePercentage)
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


