% �Ŵ��㷨�Ż�concept�����Ҷ���һ������ѵ������
clc;
clear;
close all;

%% load the data �����ݼ��ء�
load('traindata.mat');

data = traindata;
% inputdata  = [data.LowPassFiltered, data.HighPassFiltered];
% targetdata = data.OriginalData;     % Ŀ������

[seasonal, trend] = seriesDecomp(data.OriginalData, 5);

inputdata  = [seasonal, trend];
targetdata = data.OriginalData;     % Ŀ������

deletePercentage = 0.1;    % Ҫɾ�������ݱ���
numToDelete = round(725 * deletePercentage); % ����Ҫɾ��������
trainLen = 725;             % ѵ�����ݼ�����
testLen  = 365;             % �������ݼ�����
initLen  = 0;               % Ԥ�ȳ���
predictInterval = 1;        % Ԥ����

% ʱ������
traintime = data.Date(2 + initLen : trainLen);                % ѵ����ʱ��
testtime  = data.Date(trainLen + 2 : trainLen + 1 + testLen); % ���Լ�ʱ��

% ѵ������ ����[1:725] ���[1+1:725+1] ����[1:725] ���[1+5:725+5]
Xtrain = inputdata(1 : trainLen, :);                          
Ytrain = targetdata(1 + predictInterval : trainLen + predictInterval); 
indicesToDelete = randperm(size(Xtrain, 1), numToDelete); % ���ѡ��Ҫɾ����������
Xtrain(indicesToDelete, :) = [];    % ɾ����Щ��
Ytrain(indicesToDelete) = [];       % ͬ��ҲҪ��Ytrain��ɾ����Ӧ����
trainLen = 725 - numToDelete;

% �������� ����[725+1: 725 + 365] ���[725+1+1: 725 + 365 + 1]
Xtest = inputdata(trainLen + 1 : trainLen + testLen, :);
Ytest = targetdata(trainLen + 1 + predictInterval : trainLen + testLen + predictInterval);


%% generate the ESN reservoir              ������ESN�Ĵ��ز㡿
inSize  = size(inputdata,  2); % ����ڵ�����������������������ȷ��
outSize = size(targetdata, 2); % ����ڵ�����������Ŀ����������ȷ��

resSize = 100;                 % ���ڽڵ���

matrixType = 'sparse';         % ��������Ϊ 'dense' ���� 'sparse'
switch matrixType
    case 'dense'
        % �ܼ���W
        W = rand(resSize, resSize) - 0.5; % ����Ȩֵ
    case 'sparse'
        % ϡ���W
        density = 0.2; % ����ϡ�������ܶ�
        W = sprand(resSize, resSize, density); % ����ϡ����󲢵���ֵ�ķ�Χ
    otherwise
        error('δ֪�ľ������͡���ѡ�� "dense" �� "sparse".');
end

%% �Ŵ��㷨

% �Ŵ��㷨����
numWin = resSize * (1 + inSize);
numC   = resSize * resSize;
numWb  = resSize;

numVariables = numWin + numC + numWb + 2; % Win, W, Wb, a, C
lb = -1 * ones(1, numVariables); % �������½�
ub =  1 * ones(1, numVariables); % �������Ͻ�
options = optimoptions('ga', 'PopulationSize', 50, 'MaxGenerations', 10, ...
    'Display', 'iter', 'UseParallel', true);

% �����Ŵ��㷨(����IF���ѡ���Ƿ�ִ���Ŵ��㷨����ִ�о�ֱ�ӵ�������)
optimal = "GA";

if optimal == "GA"
    [optimalParams, optimalError] = ga(@(params) objectiveFunction(params, ...
        Xtrain, Ytrain, trainLen, initLen, inSize, resSize, W), ...
        numVariables, [], [], [], [], lb, ub, [], options);
    % �������
    fileName = sprintf('optimalESNParams_Interval_%d.mat', predictInterval);
    save("Parameters/"+fileName);
else
    fileName = sprintf('optimalESNParams_Interval_%d.mat', predictInterval);
    load("Parameters/"+fileName);
end
[Win, C, Wb, a, spectralRadius] = decodeParams(optimalParams, inSize, resSize);

%% normalizing and setting spectral radius
disp 'Computing spectral radius...';         % �����װ뾶
opt.disp = 0;                                % 
rhoW = abs(eigs(W,1,'LM',opt));              % �װ뾶W���������ֵ�ľ���ֵ
disp 'done.'                                 % ����
W = W .* ( spectralRadius / rhoW);           % 

% allocated memory for the design (collected states) matrix �����ڴ����
X = zeros(resSize+1, trainLen-initLen);              %  
X_origin = zeros(resSize+1, trainLen-initLen);       %
%% run the reservoir with the data and collect X
%  ���³ز��ռ�X
x = zeros(resSize, 1);                       % ��ʼ�����ز����
x_origin = zeros(resSize, 1);
Yb = 0;
for t = 1: trainLen
	u = Xtrain(t, :);                        % ����ֵ
    x_origin = (1-a)*x  + a*tanh( Win*[1 u]' + W*x + Wb' * Yb);
	x = C*x_origin;
    if t > initLen                           % ���ڳ�ʼ��ʱ
		X(:,t-initLen) = [1 x'];             % �ռ���ʼ�����[1; x; u]
        X_origin(:,t-initLen) = [1 x_origin'];
    end
    Yb = Ytrain(t);
end

%% train the output by ridge regression��ͨ����ع�ѵ�������
reg = 1e-4;
Wout = ((X*X' + reg*eye(resSize+1)) \ (X*Ytrain));  % ��ʽ27.9

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

% ����ѵ�����Ͳ������Ա�ͼ
% figure(1);
% plot(X');
% hold on;
% plot(X_origin');

% ����ѵ�����Ͳ������Ա�ͼ
fig = figure(2);
subplot(5,1,1);
plot(Ytrain, 'r');
hold on;
plot(train, 'b');
text(10, max(train), ['MSE: ' num2str(mseTrain)]); % ��ʾMSE
hold off;
legend(["Actual" "Predicted"]);
xlabel("Time Step");
ylabel("Wind Speed");
title(['ESN Training Data (Predict Interval: ' num2str(predictInterval) '): Actual vs Predicted']);

subplot(5,1,2);
hold on;
plot(Ytest, 'r');
plot(Y', 'b');
text(10, max(Ytest), ['MSE: ' num2str(mseTest)]); % ��ʾMSE
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

% �����ļ����Ļ�������
baseFilename = ['����ͼ/ESN_concept_' num2str(predictInterval)];

% ����ΪPNG��ʽ
pngFilename = [baseFilename '.png'];
saveas(fig, pngFilename);

% ����ΪFIG��ʽ��MATLABͼ�Σ�
figFilename = [baseFilename '.fig'];
saveas(fig, figFilename);

function error = objectiveFunction(params, Xtrain, Ytrain, trainLen, initLen, inSize, resSize, W)
    % ��������
    index = 0;
    Win = reshape(params(index + (1:resSize*(1+inSize))), [resSize, 1+inSize]);
    index = index + resSize*(1+inSize);
    C = reshape(params(index + (1:resSize^2)), [resSize, resSize]);
    index = index + resSize^2;
    Wb = params(index + (1:resSize));
    index = index + resSize;
    a = params(index + 1);
    spectralRadius = params(index + 2);
    
    % ����ˮ����� W ���װ뾶
    opt.disp = 0;
    rhoW = max(abs(eigs(W, 1, 'LM', opt)));
    W = W * (spectralRadius / rhoW);

    % ��ʼ��
    x = zeros(resSize, 1);                  % ��ʼ��ˮ��״̬
    X = zeros(1+resSize, trainLen-initLen); % ��ʼ���ռ�����

    % ����ESN
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

    % ѵ�����
    reg = 1e-3; % ����ϵ��
    Wout = (X*X' + reg*eye(1+resSize)) \ (X*Ytrain(initLen+1:end));

    % �������
    Ypred = Wout' * X;
    % error = mean((Ypred - Ytrain(initLen+1:end)').^2); % �������
    error = nrmse(Ypred, Ytrain(initLen+1:end)');
end

function [Win, C, Wb, a, spectralRadius] = decodeParams(optimalParams, inSize, resSize)
    % ��������
    numWin = resSize * (1 + inSize);
    numW = resSize * resSize;
    numWb = resSize;
    Win = reshape(optimalParams(1:numWin), [resSize, 1+inSize]);
    C = reshape(optimalParams(numWin+1:numWin+numW), [resSize, resSize]);
    Wb = optimalParams(numWin + numW + 1:numWin + numW + numWb);
    a = optimalParams(end - 1);
    spectralRadius = optimalParams(end); % �����װ뾶
end





