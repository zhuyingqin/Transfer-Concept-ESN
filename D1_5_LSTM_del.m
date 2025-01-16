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

inputdata1  = [data1.LowPassFiltered, data1.HighPassFiltered];
targetdata1 = data1.OriginalData;     % Ŀ������

inputdata2  = [data2.LowPassFiltered, data2.HighPassFiltered];
targetdata2 = data2.OriginalData;     % Ŀ������

deletePercentage = 0.1;     % Ҫɾ�������ݱ���
trainLen = 725;             % ѵ�����ݼ�����
testLen  = 365;             % �������ݼ�����
initLen  = 0;               % Ԥ�ȳ���
predictInterval  = 1;       % Ԥ����

[Xtrain_1, Ytrain_1, Xtest_1, Ytest_1] = processData(trainLen, testLen, ...
    predictInterval, inputdata1, targetdata1, deletePercentage, 1);

[Xtrain_2, Ytrain_2, Xtest_2, Ytest_2] = processData(trainLen, testLen, ...
    predictInterval, inputdata2, targetdata2, deletePercentage, 2);
% trainLen = size(Ytrain_1, 1);

% �������Ѿ���������ѵ�������ݣ�XTrain �� YTrain
% XTrain ���������ݣ�YTrain �Ǳ�ǩ��Ŀ���������
% ���磬XTrain ������һ��ʱ�����е���ʷ���ݣ�YTrain ����һ��ʱ�䲽��������

% 1. ���� LSTM ����ܹ�
numFeatures = size(Xtrain_1,2); % ���������������������ݵ���
numResponses = size(Ytrain_1,2); % ��Ӧ�����������������������ݵ���
numHiddenUnits = 200; % ���ز㵥Ԫ���������Ե���

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(numResponses)
    regressionLayer];

% 2. ָ��ѵ��ѡ��
options = trainingOptions('adam', ...
    'MaxEpochs',1000, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0);

% 3. ѵ�� LSTM ����
net = trainNetwork(Xtrain_1',Ytrain_1',layers,options);

% 4. ʹ��ѵ���õ��������Ԥ��
% ���� XTest �����Ĳ�������
YPred = predict(net,Xtest_1');

% 5. ����ģ������
% �Ƚ� YPred ��ʵ�ʵı�ǩ���� YTest������׼ȷ�Ȼ���������ָ��
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


