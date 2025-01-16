%%%% Morphing an integer-periodic signal to an irrational-period sinewave
%%%% H. Jaeger May 20, 2015
%%%% Designed using Matlab R2008b
%%%%
%%%% Note: I use suffix "PL" for "PlotList" in my telling-naming of
%%%% variables
%%%%
%%%% Note: users running later versions of Matlab reported that some
%%%% conceptor demos wouldn't succeed. The cause seems to be that my old
%%%% Matlab version uses a different (and better) implementation of the inv
%%%% function than later Matlab versions. Replacing all calls to "inv" by
%%%% calls to "pinv" so far has always resolved the problems. 
clc;clear;close all;
addpath('./helpers');

%% 实验参数

% 随机种子选择器，设置为任意整数
randstate = 1; 
newNets = 1; newSystemScalings = 1;
randn('state', randstate);
rand('twister', randstate);

% 设置系统参数
N = 100;                        % 网络大小
SR = 0.99;                      % 谱半径
NetinpScaling = SR;             % 模式输入权重的缩放
BiasScaling = 0.3;              % 偏置的大小

% loading 
TychonovAlpha    = 0.001;       % W训练的正则化参数
washoutLength    = 100;         % 洗出期长度
learnLength      = 5000;        % 学习期长度
signalPlotLength = 1000;        % 信号绘图长度

% readout learning
TychonovAlphaReadout = 0.0001;  % 读出权重的正则化参数

% setting the two apertures
alphas = [10 100];              % 用于Conceptor的开口参数数组

% morphing slide settings
morphWashout = 500;             % system updates before plotting starts
preMorphRecordLength = 150;     % nr of steps with first conceptor 
                                % before morphing starts
morphTime = 200;                % length of morph
delayMorphTime = 500;           % nr of delay-embedded plot points used for the 
                                % fine line in the fingerprint dot plots
delayPlotPoints = 20;           % nr of thick plot points
tN = 8;                         % nr of fingerprint panels


%% Setting patterns
periodlength = 1; % 设置周期长度

% 第一个模式
pattprofile = [0 0 3 2 5];      % 整数周期的模式概要，将会被标准化到范围 [-1 1]
pattLength  = size(pattprofile,2);
maxVal = max(pattprofile); 
minVal = min(pattprofile);
pattprofile = 2 * (pattprofile - minVal) / (maxVal - minVal) - 1;
% patts{1} = @(n) (pattprofile(mod(n, pattLength)+1));
patts{1} = @(n) sin(0.2*n/periodlength) + sin(0.311*n/periodlength);

% 第二个模式 (正弦波)
% patts{2} = @(n) sin(2 * pi * n / periodlength);
patts{2} = @(n) sin(0.2*n/periodlength) + sin(0.311*n/periodlength) + ...
            sin(0.42*n/periodlength) +  sin(0.51*n/periodlength) + ...
            sin(0.63*n/periodlength);

%% Initializations

% Create raw weights
if newNets
    if N <= 20
        Netconnectivity = 1;
    else
        Netconnectivity = 10/N;
    end
    WstarRaw = generate_internal_weights(N, Netconnectivity);
    WinRaw = randn(N, 1);
    WbiasRaw = randn(N, 1);
end

% Scale raw weights and initialize weights
if newSystemScalings
    Wstar = SR * WstarRaw;
    Win = NetinpScaling * WinRaw;
    Wbias = BiasScaling * WbiasRaw;
end

%% load patterns

Np = 2;     % 载入的模式数量，这里总是等于 2

% 初始化数据收集器
allTrainArgs    = zeros(N, Np * learnLength);
allTrainOldArgs = zeros(N, Np * learnLength);
allTrainOuts    = zeros(1, Np * learnLength);
patternRs       = cell(1, Np);

% 用不同的驱动信号驱动原生水库并收集数据
for p = 1:Np
    patt = patts{p}; % current pattern generator
    xCollector = zeros(N, learnLength );
    xOldCollector = zeros(N, learnLength );
    pCollector = zeros(1, learnLength );
    
    x = zeros(N, 1);
    for n = 1:(washoutLength + learnLength)
        u = patt(n); % pattern input
        xOld = x;
        x = tanh(Wstar * x + Win * u + Wbias);
        if n > washoutLength
            xCollector(:, n - washoutLength ) = x;
            xOldCollector(:, n - washoutLength ) = xOld;
            pCollector(1, n - washoutLength) = u;
        end
    end
    R = xCollector * xCollector' / learnLength;    
    patternRs{p} = R;    
    
    allTrainArgs(:, (p-1)*learnLength+1:p*learnLength)    = xCollector;
    allTrainOldArgs(:, (p-1)*learnLength+1:p*learnLength) = xOldCollector;
    allTrainOuts(1, (p-1)*learnLength+1:p*learnLength)    = pCollector;    
end

% 计算读出权重
L = Np * learnLength;
Wout = (inv(allTrainArgs * allTrainArgs' / L + ...
    TychonovAlphaReadout * eye(N)) * allTrainArgs * allTrainOuts' / L)';

% 训练误差
NRMSE_readout = nrmse(Wout*allTrainArgs, allTrainOuts);
fprintf('readout NRMSE: %g  mean abs size: %g\n', ...
    NRMSE_readout, mean(abs(Wout)));

% compute W
Wtargets = (atanh(allTrainArgs) - repmat(Wbias,1,Np*learnLength));
W = (inv(allTrainOldArgs * allTrainOldArgs'  / L + ...
     TychonovAlpha * eye(N)) * allTrainOldArgs * Wtargets' / L)';

% 每个神经元的训练误差
NRMSE_W = nrmse(W*allTrainOldArgs, Wtargets);
fprintf('W mean NRMSE: %g   mean abs size  %g\n', ...
    mean(NRMSE_W), mean(mean(abs(W))));

%% 计算 conceptors
Cs = cell(4, Np);
for p = 1:Np
    R = patternRs{p};
    [U S V] = svd(R);
    Snew = (S * pinv(S + alphas(p)^(-2) * eye(N)));    
    C = U * Snew * U';
    C(abs(C)<0.01) = 0;
    Cs{1, p} = C;
    Cs{2, p} = U;
    Cs{3, p} = diag(Snew);
    Cs{4, p} = diag(S);
end

%%  morphing 形态变换

ms = 0:1/morphTime:1; % 形态变换滑块规则
morphPL = zeros(1, morphTime);

% 正弦波形态变换
C1 = Cs{1,1}; C2 = Cs{1,2};
x = randn(N,1);

% 洗出期
m = ms(1);
for i = 1:morphWashout
    x = C1 * tanh(W * x + Wbias);
end

% 初始阶段，只用第一个conceptor
preMorphPL = zeros(1,preMorphRecordLength);
m = ms(1);
for i = 1:preMorphRecordLength
    x = C1 * tanh(W * x + Wbias);
    preMorphPL(1,i) = Wout * x;
end

% 形态变换滑块
for i = 1:morphTime
    m = ms(i);
    x = ((1-m)*C1 + m*C2) * tanh(W * x + Wbias);
    morphPL(1,i) = Wout * x;
end

% 形态变换后
postMorphRecordLength = preMorphRecordLength;
postMorphPL = zeros(1,postMorphRecordLength);
for i = 1:postMorphRecordLength
    x = C2 * tanh(W * x + Wbias);
    postMorphPL(1,i) = Wout * x;    
end
% 组合三个部分的模式输出
totalMorphPL = [preMorphPL morphPL postMorphPL];

% 插值（用于底部面板中周期长度估计）
interpolInc = 0.1;
L = preMorphRecordLength + morphTime + postMorphRecordLength;
interpolPoints = 1:interpolInc:L;
interpolL = length(interpolPoints);
totalMorphPLInt = interp1((1:L)', totalMorphPL', interpolPoints', 'spline');

% 计算参考周期长度，显示为底部面板中的细线
p1 = pattLength; p2 = periodlength;
pdiff = p2 - p1;
pstart = p1; pend = p1 + pdiff;
refPL = [pstart * ones(1,preMorphRecordLength), ...
    (0:(morphTime-1)) / (morphTime-1) * (pend-pstart)+pstart,...
    pend * ones(1,preMorphRecordLength)];

% 从插值的峰值距离估计周期长度
crestDistcounts = zeros(1, interpolL);
oldVal = 1;
counter = 0;
for i = 1:interpolL-2
    if totalMorphPLInt(i+1) - totalMorphPLInt(i) > 0 && ...
            totalMorphPLInt(i+2) - totalMorphPLInt(i+1) <= 0 &&...
            totalMorphPLInt(i) > .5
        counter = counter + 1;
        crestDistcounts(i) = counter;
        oldVal = counter;
        counter = 0;
    else
        crestDistcounts(i) = oldVal;
        counter = counter + 1;
    end
end

% 子采样
crestDistcounts = crestDistcounts(1,interpolInc^(-1):interpolInc^(-1):interpolL);
crestDistcounts = crestDistcounts * interpolInc;

% 填补开始
crestDistcounts(1,1:20) = ones(1,20) * crestDistcounts(20);

% 填补结束
crestDistcounts = [crestDistcounts(1,1:end-5), ...
    ones(1,6) * crestDistcounts(end-5)];

% 计算延迟绘图指纹
slide1 = 0 : 1/(tN-1) : 1;
delayData = zeros(tN, delayMorphTime);
x0 = rand(N,1);
for i = 1:tN
    x = x0;
    Cmix = (1-slide1(i)) * C1 + slide1(i) * C2;
    for n = 1:morphWashout
        x = Cmix * tanh(W * x + Wbias);
    end
    for n = 1:delayMorphTime
        x = Cmix * tanh(W * x + Wbias);
        delayData(i,n) = Wout * x;
    end
end
fingerPrintPoints = preMorphRecordLength + (0:tN-1)*morphTime/(tN-1);

%% 绘图
figure(1); clf;
fs = 10; % 字体大小
lw = 1.5;  % 线宽
set(gcf, 'WindowStyle','normal');
set(gcf,'Position', [900 400 800 400]);

% 颜色设置
c = 0.9; col1 = [1 c c]; col2 = [c c 1];
col1a = [1 0 0]; col2a = [0 0 1];

% 子图1: 循环延迟图 
% 第一个用于展示整个时间序列的动态行为，而第二个则用于强调序列中的特定部分。
for i = 1:tN
    panelWidth = (1/(tN + 1)) * (1-0.08);
    panelHight = 1/(3.5);
    panelx = (1-0.08)*(i-1)*(1/tN) + ...
        (1-0.08)*(i-1)*(1/tN - panelWidth)/tN...
        + 0.04;
    panely = 2/3 ;
    
    subplot('Position', [panelx, panely, panelWidth, panelHight]);
    thisdata = delayData(i,1:delayPlotPoints+1);
    plot(delayData(i, 1:end-1), delayData(i, 2:end),'ko',...
         'MarkerSize',2, 'MarkerFaceColor', 'k');
    hold on;
    colors = thisdata(1,1:end-1);
    scatter(thisdata(1,1:end-1), thisdata(1,2:end), 25, colors, 'filled');
    colormap(parula); % 使用jet颜色映射来显示渐变效果
    % plot(thisdata(1,1:end-1), thisdata(1,2:end), 'k.', 'MarkerSize',25);
    hold off;
    set(gca, 'XTickLabel',[],'YTickLabel',[],...
        'XLim', [-3 1.4], 'YLim',[-3 1.4], 'Box', 'on', ...
        'Color', (1-slide1(i))*col1 + slide1(i)*col2);
end

% 子图2: 形态变化时间序列
morphStart = preMorphRecordLength;
morphEnd   = preMorphRecordLength + morphTime;

subplot('Position',[0.04 0.1 1-0.08 0.5]);
hold on;
line([morphStart morphStart], [-4 2],...
    'Color', 'k', 'LineWidth', 1.2);
line([morphEnd morphEnd], [-4 2],...
    'Color', 'k', 'LineWidth', 1.2);

plot(totalMorphPL(1,1:morphStart), 'Color', col1a, 'LineWidth', lw);
% 添加公式
text(30, -3.5, '$$ \sin(0.2x) + \sin(0.311x) $$', 'Interpreter', 'latex');
text(55, 2.3,  'Modo 1');
text(425, 2.3, 'Modo 2');
slide2 = (0:(morphEnd-morphStart-1))/((morphEnd-morphStart-1));
for i = morphStart:(morphEnd-1)
    plot([i,i+1], totalMorphPL(1,i:i+1), ...
        'Color', (1-slide2(i-morphStart+1))*col1a + ...
        slide2(i-morphStart+1)*col2a, 'LineWidth', lw);
end
text(140, 2.3, '$$ a=0 $$', 'Interpreter', 'latex');
text(340, 2.3, '$$ a=1 $$', 'Interpreter', 'latex');
plot(morphEnd:(length(totalMorphPL)), ...
    totalMorphPL(1,morphEnd:end), 'Color', col2a, 'LineWidth', lw);
text(355, -3.0, '$$ \sin(0.2x) + \sin(0.311x) + \sin(0.42x) $$', ...
    'Interpreter', 'latex');

text(355, -3.5, '$$ + \sin(0.51x) + \sin(0.63x) $$', 'Interpreter', 'latex');

% plot([morphStart, morphEnd], [-1 -1], 'k.', 'MarkerSize',35);

% 绘制图中的三角形
plot(fingerPrintPoints, 2*ones(1,tN), 'kv', 'MarkerSize',10, ...
    'MarkerFaceColor','k');
hold off;

set(gca, 'FontSize',fs, 'Box', 'on');

% 子图3: 周期长度估计
% subplot('Position',[0.04 0.03 1-0.08 1/3-0.05]);
% hold on;
% line([morphStart morphStart], [4 10],...
%     'Color', 'k', 'LineWidth', 2);
% line([morphEnd morphEnd], [4 10],...
%     'Color', 'k', 'LineWidth', 2);
% plot(crestDistcounts, 'LineWidth',6,'Color',0.75*[1 1 1]);
% plot(refPL, 'k-');
% hold off;
% set(gca, 'YLim', [4 ceil(periodlength+1)],...
%     'XTickLabel',[], 'Box','on', 'FontSize',fs);





