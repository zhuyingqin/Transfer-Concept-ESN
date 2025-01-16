% �Ŵ��㷨�Ż�concept�����Ҷ���һ������ѵ������
clc;
clear;
close all;
% parpool(4)

%% load the data �����ݼ��ء�
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

inputdata1  = abs(fft([data1.LowPassFiltered, data1.HighPassFiltered]));
targetdata1 = data1.OriginalData;     % Ŀ������

inputdata2  = abs(fft([data2.LowPassFiltered, data2.HighPassFiltered]));
targetdata2 = data2.OriginalData;     % Ŀ������

deletePercentage = 0;     % Ҫɾ�������ݱ���
trainLen = 725;             % ѵ�����ݼ�����
testLen  = 365;             % �������ݼ�����
initLen  = 0;               % Ԥ�ȳ���
predictInterval = 4;        % Ԥ����

[Xtrain_1, Ytrain_1, Xtest_1, Ytest_1] = processData(trainLen, testLen, ...
    predictInterval, inputdata1, targetdata1, deletePercentage, 1);

[Xtrain_2, Ytrain_2, Xtest_2, Ytest_2] = processData(trainLen, testLen, ...
    predictInterval, inputdata2, targetdata2, deletePercentage, 2);
trainLen = size(Ytrain_1, 1);


%% ����ESN�Ĵ���
inSize  = size(inputdata1,  2); % ����ڵ�����������������������ȷ��
outSize = size(targetdata1, 2); % ����ڵ�����������Ŀ����������ȷ��

resSize = 100;                   % ���ڽڵ���

Win = (rand(resSize, 1+inSize) - 1) .* 1;    % ��ʼ������
Wb  = rand(resSize, 1);
matrixType = 'sparse';          % ��������Ϊ 'dense' ���� 'sparse'
switch matrixType
    case 'dense'
        % �ܼ��� W
        W = rand(resSize, resSize) - 0.5;    % ����Ȩֵ
    case 'sparse'
        % ϡ��� W
        density = 0.2; % ����ϡ�������ܶ�
        W = sprand(resSize, resSize, density);
    otherwise
        error('δ֪�ľ������͡���ѡ�� "dense" �� "sparse".');
end

% ����ESN�ṹ��[�̶�����]
ESN1 = buildESNParameters(resSize, initLen, trainLen, testLen, matrixType, ...
        density, W, Win, Wb, Xtrain_1, Ytrain_1, Xtest_1, Ytest_1, inSize, outSize);

% ESN2 = buildESNParameters(resSize, initLen, trainLen, testLen, matrixType, ...
%         density, W, Win, Wb, Xtrain_2, Ytrain_2, Xtest_2, Ytest_2, inSize, outSize);

%% �Ŵ��㷨

% �����Ŵ��㷨(����IF���ѡ���Ƿ�ִ���Ŵ��㷨����ִ�о�ֱ�ӵ�������)
optimal = "GA";

if optimal == "GA"
    numC   = 0;             % 
    numVariables = numC + 2;                % C1 C2 Row
    lb = -1 * ones(1, numVariables);        % �������½�
    ub =  1 * ones(1, numVariables);        % �������Ͻ�
    options = optimoptions('ga', 'PopulationSize', 75, 'MaxGenerations', 200, ...
    'Display', 'iter', 'UseParallel', true, 'FunctionTolerance', 1e-8, ...
    'MutationFcn', {@mutationuniform, 0.2}, ...
    'PlotFcn', @(options, state, flag) gaoutfun(options, state, flag, ESN1));
    [optimalParams, optimalError1] = ga(@(params) objectiveFunction(params, ...
    ESN1), numVariables, [], [], [], [], lb, ub, [], options);
    % �������
    fileName = sprintf('ESNParams_Interval_del02_%d.mat', index);
    save("Parameters/"+fileName);

else
    fileName = sprintf('ESNParams_Interval_del02_%d.mat', index);
    load("Parameters/"+fileName);
    %ESN1.a = 1;
    %ESN2.a = 1 - ESN1.a;    
end

%% normalizing and setting spectral radius
[spectralRadius, a] = decodeParams(optimalParams, resSize);
ESN1.a = abs(a);


disp 'Computing spectral radius...';         % �����װ뾶
opt.disp = 0;                                % 
rhoW1 = abs(eigs(ESN1.W,1,'LM',opt));              
disp 'done.'                                 % ����
ESN1.Wres = ESN1.W .* ( spectralRadius / rhoW1);   % 

%% ͨ����ع�ѵ�����
X = computeStateMatrix(ESN1, "train");   % ����ѵ����״̬����

reg = 0.001;
Wout = ((X*X' + reg*eye(resSize)) \ (X*ESN1.Ytrain));    % �������Ȩ��

train = Wout' * X;

X_test = computeStateMatrix(ESN1, "test", X(:, end));
Y = Wout' * X_test;

mseTest   = Mse(Y(50:end), ESN1.Ytest(50:end)')
Rv = calculateRSquared(Y(50:end), ESN1.Ytest(50:end)')

%% ����ѵ�����Ͳ������Ա�ͼ
fig = figure(2);
subplot(4,1,1);
plot(ESN1.Ytrain, 'r');
hold on;
plot(train, 'b');
hold off;
legend(["Actual" "Predicted"]);
xlabel("Time Step");
ylabel("Wind Speed");
title(['ESN Training Data (Predict Interval: ' num2str(index) '): Actual vs Predicted']);

subplot(4,1,2);
plot(ESN1.Ytest(10:end), 'r');
hold on;
plot(Y(10:end)', 'b');
text(10, max(ESN1.Ytest), ['MSE: ' num2str(mseTest)]); % ��ʾMSE
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


function error = objectiveFunction(params, ESN1)
    % ��������
    [spectralRadius, a] = decodeParams(params, ESN1.resSize);

    % ����ˮ����� W ���װ뾶
    opt.disp = 0;
    rhoW1 = max(abs(eigs(ESN1.W, 1, 'LM', opt)));
    ESN1.Wres = ESN1.W * (spectralRadius / rhoW1);
    ESN1.a    = abs(a);
    
    X = computeStateMatrix(ESN1, "train", zeros(ESN1.resSize, 1));
    
    % ѵ�����
    reg = 1e-1; % ����ϵ��
    Wout = (X*X' + reg*eye(ESN1.resSize)) \ (X*ESN1.Ytrain(ESN1.initLen+1:end));

    % �������
    Ypred = Wout' * X;
    error = mean((Ypred - ESN1.Ytrain(ESN1.initLen+1:end)').^2); % �������
    % error = nrmse(Ypred, Ytrain(initLen+1:end)');

    % error = tesFitnessFunction(params, ESN1, ESN2);
end

function [spectralRadius, a] = decodeParams(params, resSize)
    % ��������
    index = 0;
    spectralRadius = params(index + 1);
    a = params(index + 2);
end

function ESN = buildESNParameters(resSize, initLen, trainLen, testLen, ...
    matrixType, density, W, Win, Wb, Xtrain, Ytrain, Xtest, Ytest, ...
    inSize, outSize)

    % ��ʼ��ESN�ṹ��
    ESN.inSize  = inSize;    % ����ߴ�
    ESN.outSize = outSize;   % ����ߴ�

    % ����ESN����
    ESN.resSize = resSize;              % ���ڽڵ���
    ESN.initLen = initLen;              % ��ʼ������
    ESN.trainLen = trainLen;            % ѵ������
    ESN.testLen = testLen;              % ���Գ���
    ESN.matrixType = matrixType;        % ��������: 'dense' �� 'sparse'
    ESN.density = density;              % ϡ�������ܶ�
    ESN.W       = W;
    ESN.Win     = Win;
    ESN.Wb      = Wb;
    ESN.Xtrain  = Xtrain;
    ESN.Ytrain  = Ytrain;
    ESN.Xtest   = Xtest;
    ESN.Ytest   = Ytest;
end

function X = computeStateMatrix(ESN, type, lastone)
    % �ӽṹ������ȡ����
    resSize  = ESN.resSize;
    initLen  = ESN.initLen;
    trainLen = ESN.trainLen;
    testLen  = ESN.testLen;
    Win      = ESN.Win;
    Wres     = ESN.Wres;
    Wb       = ESN.Wb;
    
    % ��ʼ�����ز�״̬����
    x  = zeros(resSize, 1);
    if type == "test"
        Len = testLen;
        X = zeros(resSize, testLen - initLen);
        Input    = ESN.Xtest;
        Output   = ESN.Ytest;
        Yb       = ESN.Ytrain(end);
        x        = lastone;

    elseif type == "train"
        Len = trainLen;
        X = zeros(resSize, trainLen - initLen);
        Input    = ESN.Xtrain;
        Output   = ESN.Ytrain;
        Yb       = 0;
    end

    % ���д��ز㲢�ռ�״̬
    for t = 1:Len
        u  = Input(t, :); 
        
        x  = tanh(Win * [1 u]'  + Wres * x  + Wb*Yb);
   
        % x  = ESN.a * ESN.C  * x;
        if t > initLen
            X(:, t - initLen) = x;
        end
        Yb  = Output(t);
    end

end

function [Xtrain, Ytrain, Xtest, Ytest] = processData(trainLen, testLen, predictInterval, inputdata, targetdata, deletePercentage,seedd)
    rng(seedd);
    % ѵ������ ����[1:725] ���[1+10:725+10]
    Xtrain = inputdata(1 : trainLen, :);                          
    Ytrain = targetdata(1 + predictInterval : trainLen + predictInterval); 

    % �������� ����[725+1: 725 + 365] ���[725+1+10: 725 + 365 + 10]
    Xtest = inputdata(trainLen + 1 : trainLen + testLen, :);
    Ytest = targetdata(trainLen + 1 + predictInterval : trainLen + testLen + predictInterval);
    
    numToDelete = round(trainLen * deletePercentage);   % ����Ҫɾ��������
    indicesToDelete = randperm(size(Xtrain, 1), numToDelete); % ���ѡ��Ҫɾ����������
    Xtrain(indicesToDelete, :) = [];    % ɾ����Щ��
    Ytrain(indicesToDelete) = [];       % ͬ��ҲҪ��Ytrain��ɾ����Ӧ����
end

function state = gaoutfun(~,state,flag, ESN1)
    persistent bestscores testScores
    if isempty(bestscores) || strcmp(flag,'init')
        bestscores = [];
        testScores = [];
    end
    
    % ÿ10����¼һ������
    if mod(state.Generation, 10) == 0
        if strcmp(flag, 'iter')
            % ʹ��state.Scoreȷ�����Ÿ���
            [bestScore, bestIndex] = min(state.Score); % ��������С������
            bestIndividual = state.Population(bestIndex,:);
            
            % ��¼ѵ���������Ӧ��
            bestscores = [bestscores; bestScore];
            
            % �ڲ��Լ�������������Ÿ���
            testScore = tesFitnessFunction(bestIndividual,ESN1); 
            testScores = [testScores; testScore];
            
            % ����ѵ�����Ͳ��Լ���Ӧ��
            yyaxis left
            plot(bestscores, 'LineWidth', 1.2, ...
                        'Marker','o','LineStyle','--');
            ylabel('Training error');
            yyaxis right
            plot(testScores, 'LineWidth', 1.2, ...
                        'Marker','x','LineStyle','-.');
            xlabel('Generation');
            ylabel('Testing error');
            drawnow
        end
    end
    
    if strcmp(flag,'done') % �����ʱ����
        clear bestscores testScores
    end
end

function error = tesFitnessFunction(bestIndividual, ESN1)
    % ��������
    [spectralRadius, a] = decodeParams(bestIndividual, ESN1.resSize);
    % ����ˮ����� W ���װ뾶
    opt.disp = 0;
    rhoW1 = max(abs(eigs(ESN1.W, 1, 'LM', opt)));
    ESN1.Wres = ESN1.W * (spectralRadius / rhoW1);
    ESN1.a    = abs(a);
        
    X = computeStateMatrix(ESN1, "train", zeros(ESN1.resSize, 1));

    % ѵ�����
    reg = 1e-1; % ����ϵ��
    Wout = (X*X' + reg*eye(ESN1.resSize)) \ (X*ESN1.Ytrain(ESN1.initLen+1:end));
    
    X_test = computeStateMatrix(ESN1, "test", zeros(ESN1.resSize, 1));

    Y = Wout' * X_test;

    error  = Mse(Y(50:end), ESN1.Ytest(50:end)');
    % error  = rmse(Y(50:end), ESN1.Ytest(50:end)')
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

