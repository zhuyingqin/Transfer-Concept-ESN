% �Ŵ��㷨�Ż�concept�����Ҷ���һ������ѵ������
clc;
clear;
close all;

%% load the data �����ݼ��ء�
wp1 = load('traindata.mat');
wp2 = load('traindata3.mat');

data1 = wp1.traindata;
data2 = wp2.traindata;

inputdata1  = [data1.LowPassFiltered, data1.HighPassFiltered];
targetdata1 = data1.OriginalData;     % Ŀ������

inputdata2  = [data2.LowPassFiltered, data2.HighPassFiltered];
targetdata2 = data2.OriginalData;     % Ŀ������

% [bestShift, Data2_aligned]   = findBestLagViaCrossCorrelation(inputdata1, inputdata2);
% [bestShift, target2_aligned] = findBestLagViaCrossCorrelation(targetdata1, targetdata2);
% inputdata2  = Data2_aligned;
% targetdata2 = target2_aligned;

deletePercentage = 0.2;     % Ҫɾ�������ݱ���
trainLen = 725;             % ѵ�����ݼ�����
testLen  = 365;             % �������ݼ�����
initLen  = 0;               % Ԥ�ȳ���
predictInterval = 3;        % Ԥ����

[Xtrain_1, Ytrain_1, Xtest_1, Ytest_1] = processData(trainLen, testLen, ...
    predictInterval, inputdata1, targetdata1, deletePercentage);

[Xtrain_2, Ytrain_2, Xtest_2, Ytest_2] = processData(trainLen, testLen, ...
    predictInterval, inputdata2, targetdata2, deletePercentage);
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
        % �ܼ���W
        W = rand(resSize, resSize) - 0.5;    % ����Ȩֵ
    case 'sparse'
        % ϡ���W
        density = 0.2; % ����ϡ�������ܶ�
        W = sprand(resSize, resSize, density);
    otherwise
        error('δ֪�ľ������͡���ѡ�� "dense" �� "sparse".');
end

% ����ESN�ṹ��[�̶�����]
ESN1 = buildESNParameters(resSize, initLen, trainLen, testLen, matrixType, ...
        density, W, Win, Wb, Xtrain_1, Ytrain_1, Xtest_1, Ytest_1, inSize, outSize);

ESN2 = buildESNParameters(resSize, initLen, trainLen, testLen, matrixType, ...
        density, W, Win, Wb, Xtrain_2, Ytrain_2, Xtest_2, Ytest_2, inSize, outSize);

%% �Ŵ��㷨

% �����Ŵ��㷨(����IF���ѡ���Ƿ�ִ���Ŵ��㷨����ִ�о�ֱ�ӵ�������)
optimal = "GA";

if optimal == "GA"
    ESN1.a = 0.5;
    ESN2.a = 0.5;
    numC   = resSize * resSize * 2;
    numVariables = numC + 1; % C1 C2 Row
    lb = -1 * ones(1, numVariables);  % �������½�
    ub =  1 * ones(1, numVariables);  % �������Ͻ�
    options = optimoptions('ga', 'PopulationSize', 75, 'MaxGenerations', 2000, ...
    'Display', 'iter', 'UseParallel', true, 'FunctionTolerance', 1e-8, ...
    'MutationFcn', {@mutationuniform, 0.2});
    [optimalParams, optimalError1] = ga(@(params) objectiveFunction(params, ...
    ESN1, ESN2), numVariables, [], [], [], [], lb, ub, [], options);
    % �������
    fileName = sprintf('optimalESNParams_Interval_del_%d.mat', predictInterval);
    save("Parameters/"+fileName);
else
    fileName = sprintf('optimalESNParams_Interval_del_%d.mat', predictInterval);
    load("Parameters/"+fileName);
end
ESN1.a = 0.8;
ESN2.a = 1 - ESN1.a;
%% normalizing and setting spectral radius
[ESN1.C, ESN2.C, spectralRadius] = decodeParams(optimalParams, resSize);

disp 'Computing spectral radius...';         % �����װ뾶
opt.disp = 0;                                % 
rhoW = abs(eigs(W,1,'LM',opt));              % �װ뾶W���������ֵ�ľ���ֵ
disp 'done.'                                 % ����
ESN1.Wres = W .* ( spectralRadius / rhoW);   % 
ESN2.Wres = W .* ( spectralRadius / rhoW);   % 

% �����ݾ��� X ���� SVD �ֽ�
[U1, S1, V1] = svd(ESN1.C, 'econ'); % ʹ�� 'econ' ѡ����о����ͷֽ�
[U2, S2, V2] = svd(ESN2.C, 'econ'); % ʹ�� 'econ' ѡ����о����ͷֽ�
%% ͨ����ع�ѵ�����
X = computeStateMatrix(ESN1, ESN2, "train");   % ����ѵ����״̬����

reg = 1e-3;
Wout = ((X*X' + reg*eye(resSize)) \ (X*ESN1.Ytrain));    % �������Ȩ��

train = Wout' * X;

X_test = computeStateMatrix(ESN1, ESN2, "test");
Y = Wout' * X_test;

mseTest  = Mse(Y(50:end), ESN1.Ytest(50:end)');
mseTrain = Mse(train, ESN1.Ytrain');

%% ����ѵ�����Ͳ������Ա�ͼ
fig = figure(2);
subplot(4,1,1);
plot(ESN1.Ytrain, 'r');
hold on;
plot(train, 'b');
text(10, max(train), ['MSE: ' num2str(mseTrain)]); % ��ʾMSE
hold off;
legend(["Actual" "Predicted"]);
xlabel("Time Step");
ylabel("Wind Speed");
title(['ESN Training Data (Predict Interval: ' num2str(predictInterval) '): Actual vs Predicted']);

subplot(4,1,2);
hold on;
plot(ESN1.Ytest, 'r');
plot(Y', 'b');
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

% �����ļ����Ļ�������
baseFilename = ['����ͼ/ESN_concept_' num2str(predictInterval)];

% ����ΪPNG��ʽ
pngFilename = [baseFilename '.png'];
saveas(fig, pngFilename);

% ����ΪFIG��ʽ��MATLABͼ�Σ�
figFilename = [baseFilename '.fig'];
saveas(fig, figFilename);

function error = objectiveFunction(params, ESN1, ESN2)
    % ��������
    [C1, C2, spectralRadius] = decodeParams(params, ESN1.resSize);

    % ����ˮ����� W ���װ뾶
    opt.disp = 0;
    rhoW = max(abs(eigs(ESN1.W, 1, 'LM', opt)));
    ESN1.Wres = ESN1.W * (spectralRadius / rhoW);
    ESN1.C    = C1;

    ESN2.Wres = ESN2.W * (spectralRadius / rhoW);
    ESN2.C    = C2;
    
    X = computeStateMatrix(ESN1, ESN2, "train");
    
    % ѵ�����
    reg = 1e-3; % ����ϵ��
    Wout = (X*X' + reg*eye(ESN1.resSize)) \ (X*ESN1.Ytrain(ESN1.initLen+1:end));

    % �������
    Ypred = Wout' * X;
    error = mean((Ypred - ESN1.Ytrain(ESN1.initLen+1:end)').^2); % �������
    % error = nrmse(Ypred, Ytrain(initLen+1:end)');
end

function [C1, C2, spectralRadius] = decodeParams(params, resSize)
    % ��������
    index = 0;
    C1 = reshape(params(index + (1:resSize^2)), [resSize, resSize]);
    
    index = index + resSize^2;
    C2 = reshape(params(index + (1:resSize^2)), [resSize, resSize]);
    
    index = index + resSize^2;
    spectralRadius = params(index + 1);
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

function X = computeStateMatrix(ESN, ESN2, type)
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

    % ���д��ز㲢�ռ�״̬
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


