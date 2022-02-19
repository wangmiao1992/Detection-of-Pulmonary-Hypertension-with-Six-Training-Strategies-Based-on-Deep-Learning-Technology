clc
clear all
close all

%% 未分割片段
load ('E:\2021.1.18.临床数据处理\Labels.mat');
load ('E:\2021.1.18.临床数据处理\Signals.mat');

HS=[];
for i=1:118
    
    HS1=Signals{i, 1}(:,4);
    HS1=HS1(1:122633,1);
    HS=[HS HS1];
    
end

for i=119:131
    
    HS1=Signals{i, 1}(:,2);
    HS1=HS1(1:122633,1);
    HS=[HS HS1];
    
end
HS=HS';
label_dua=Labels;
label_dua=cell2mat(label_dua);

%% 心电和心音带通滤波器去噪之后并归一化；
fs=1000;
f1=20; %cuttoff low frequency to get rid of baseline wander
f2=200; %cuttoff frequency to discard high frequency noise
Wn=[f1 f2]*2/fs; % cutt off based on fs
N = 3; % order of 3 less processing
[a,b] = butter(N,Wn); %bandpass filtering

HSS=[];
for i=1:length(HS(:,1))
HS_all1 = filtfilt(a,b,HS(i,:));
HS_all1 = HS_all1/ max( abs(HS_all1));%归一化
HSS=[HSS 
    HS_all1];
i
end
HS=HSS;
%% 将四分类转换为2分类；
a=find( label_dua>=1);
label_dua(a,1)=1;

%% 导入归一化后的分割数据，每隔10s一个记录，总共5164个记录，整体保存，按照单独的记录保存的；
%% 此数据可用于按照打散混合的的方式训练；
% load('E:\2021.3.5\normalization.mat');

%% 分段后，单独按照sublect_independent保存归一化并去噪后的数据；一共131个个体，总共5164个记录；
%% 此数据可用于按照个体间不打散的方式训练；
% load('E:\2021.3.5\2021.4.1.重新分割数据\segments.mat');
% HS=segments;

% %% 频率插值后，提取MFCC特征；
% HS2=HS(1,:)';
% HS2=interp(HS2,5);
% Fs=5000;
% [coeffs,delta,deltaDelta,loc] = mfcc(HS2,Fs);
%% 新建一个.wav，读入.wav后，再读出。
% filename = 'E:\2021.1.28.音频数据\test_HS.wav';
% audiowrite(filename,HS,Fs);
% clear HS Fs
% [HS,Fs] = audioread(filename);
% heatmap(coeffs(:,1))


%% 小波散射
%% 心音
Fs=1000;
sampleSig=HS(1,:);
sf = waveletScattering('SignalLength',numel(sampleSig),'SamplingFrequency',Fs)
feat = featureMatrix(sf,sampleSig);

lev = 1;
[S0,U0] = scatteringTransform(sf,HS(find(label_dua==0,1),:));
[S1,U1] = scatteringTransform(sf,HS(find(label_dua==1,1),:));
% [S2,U2] = scatteringTransform(sf,HS(find(label_dua==2,1),:));
% [S3,U3] = scatteringTransform(sf,HS(find(label_dua==3,1),:));
figure;
scattergram(sf,S0,'FilterBank',lev);
figure;
scattergram(sf,S1,'FilterBank',lev);
% figure;
% scattergram(sf,S2,'FilterBank',lev);
% figure;
% scattergram(sf,S3,'FilterBank',lev);

% %% 让我们将数据集随机划分为训练数据集和测试数据集。
% M = size(HS, 1);
% idxsel = randperm(M, 4);
% tiledlayout(2, 2, "Padding", "compact")
% 
% idxRandomized = randperm(M);
% trainSize = round(M*0.7);
% testSize = M-trainSize;
% trainData = zeros(trainSize, size(HS,2));
% trainLabels = cell(trainSize,1);
% testData = zeros(testSize, size(HS,2));
% testLabels = cell(testSize,1);

% for i=1:M
%     if i <= trainSize
%         trainData(i,:) = HS(idxRandomized(i),:);
%         trainLabels{i,1} = label_dua(idxRandomized(i));
%     else
%         testData(i-trainSize,:) = HS(idxRandomized(i),:);
%         testLabels{i-trainSize,1} = label_dua(idxRandomized(i));
%     end
%     
% end


%% 让我们将数据集随机划分为训练数据集和测试数据集。
indices = crossvalind('Kfold',label_dua,5);
    for i = 1:5 %循环5次，分别取出第i部分作为测试样本，其余两部分作为训练样本
        test = (indices == i);
        train = ~test;
        trainData = HS(train, :);
        trainLabels = label_dua(train, :);
        testData = HS(test, :);
        testLabels = label_dua(test, :);
        
        trainSize=length(trainLabels);
        testSize=length(testLabels);
        %% 五倍交叉验证五次结果；
        %% 使用我们现有的小波散射变换，为训练和测试数据集计算散射特征。
scat_features_train = cell(trainSize,1);
scat_features_test = cell(testSize,1);

parfor itr = 1:trainSize   % Training Records
    tmp = featureMatrix(sf,trainData(itr,:));
    scat_features_train{itr,1} =tmp;
end

parfor ttr = 1:testSize    %  Test Records
    tmp = featureMatrix(sf,testData(ttr,:));
    scat_features_test{ttr,1} =tmp;
end

[inputSize,~] = size(scat_features_train{1});

%% 转换输入的格式；
% trainLabels=cell2mat(trainLabels);
A = categorical(trainLabels,[0 1],{'health' 'unhealth'});
YTrain = categorical(A);

numHiddenUnits = 100;
numClasses = numel(unique(YTrain));
maxEpochs = 125;
miniBatchSize = 1000;

layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'InitialLearnRate',0.01,...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','shortest', ...
    'Shuffle','never',...
    'Plots','training-progress');

netScat = trainNetwork(scat_features_train,YTrain,layers,options);

% testLabels=cell2mat(testLabels);
B = categorical(testLabels,[0 1],{'health' 'unhealth'});
YTest = categorical(B);

YPred = classify(netScat,scat_features_test, 'MiniBatchSize',miniBatchSize, 'SequenceLength','shortest');
accuracy = round((sum(YPred == YTest)./numel(YTest))*100);
acc(i,:)=accuracy;

figure;
confusionchart(YTest, YPred, "RowSummary", "row-normalized");
title("Accuracy: " + accuracy + "%")
     
        
    end


