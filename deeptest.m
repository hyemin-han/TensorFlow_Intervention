clear all;

% load test data
M = csvread("test1.csv");

% delete id
Newdata = M(:,2:11);


% set aside test data
Fulldata = Newdata;
Newdata = Newdata(1:size(Newdata,1)-30,:);
leng = size(Newdata,1);

% create label and classify
count_o = 1;
count_x = 1;
for i=1:leng
    % participate in T2?
    if Newdata(i,7)
        % yes
        PO (1:6,count_o) = Newdata(i,1:6);
        PO (7:9,count_o) = Newdata(i,8:10);
        count_o = count_o + 1;
    else
        % no
        PX (1:6,count_x) = Newdata(i,1:6);
        PX (7:9,count_x) = Newdata(i,8:10);
        count_x = count_x + 1;
    end
end
% create a cell array
Data{1,1} = PO;
Data{2,1} = PX;
% Labels ={'1','0'};
Labels ={'1';'0'};
Labels = categorical(Labels);

% get the sequence length for obs
numObservations = numel(Data);
for i=1:numObservations
    sequence = Data{i};
    sequenceLengths(i) = size(sequence,2);
end

% sort data
[sequenceLengths,idx] = sort(sequenceLengths);
Data = Data(idx);
Labels = Labels(idx);

% setting options
% 9 variables
inputSize = 9;
% 100 hidden units
numHiddenUnits = 100;
% 2 classified outputs
numClasses = 2;

% create layers
layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

maxEpochs = 100;
miniBatchSize = 27;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(Data,Labels,layers,options);


% Test dataset
clear TestData;
clear TData;
clear TLabels;
clear TPLabels;
clear TTData;
TestData = Fulldata(size(Newdata,1)-30+1:size(Newdata,1),:);
leng = size(TestData,1);

% create label and classify
count_o = 1;
count_x = 1;
for i=1:leng
    % participate in T2?
    if Newdata(i,7)
        % yes
        TPO (1:6,count_o) = TestData(i,1:6);
        TPO (7:9,count_o) = TestData(i,8:10);
        count_o = count_o + 1;
    else
        % no
        TPX (1:6,count_x) = TestData(i,1:6);
        TPX (7:9,count_x) = TestData(i,8:10);
        count_x = count_x + 1;
    end
    TTData (1:6, i) = TestData(i,1:6);
    TTData(7:9, i) = TestData(i,8:10);
end

for i=1:leng
    TData{i,1} = TTData(:,i);
end

% create a cell array
% TData{1,1} = TPO;
% TData{2,1} = TPX;
% create label list
for i=1:leng
    % Labels ={'1','0'};
%     TLabels ={'1';'0'};
%     TLabels = categorical(TLabels);
    if TestData(i,7) == 1
        TLabels{i} = '1';
    else
        TLabels{i} = '0';
    end
end
% extract and sort data
% numObservationsTest = numel(TData);
% clear idx;
% for i=1:numObservationsTest
%     sequence = TData(i);
%     sequenceLengthsTest(i) = size(sequence,2);
% end
% [sequenceLengthsTest,idx] = sort(sequenceLengthsTest);
% TData = TData(idx);
% TLabels = TLabels(idx);

% Test!
miniBatchSize = 15;
TPLabels = classify(net,TData, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest');