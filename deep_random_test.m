clear all;

% load test data
M = csvread("test1.csv");
% delete id
Newdata = M(:,2:11);
% set aside test data
Fulldata = Newdata;
%Newdata = Newdata(1:size(Newdata,1)-30,:);
leng = size(Newdata,1);

clearvars -except trials M Newdata Fulldata leng Acc_Val  Acc_Trn;

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

% Set aside 30 validation sets
RAND = rand([leng 1]);

% ranking
[~,p] = sort(RAND,'descend');
r = 1:length(RAND);
r(p) = r;

% Data selection.
valdata = 1;
traindata = 1;
tcount_o = 1;
tcount_x = 1;
tcount_o = 1;
tcount_x = 1;
for i=1:leng
    % for validation
    if (0)
    % participate in T2?
        if Newdata(i,7)
            % yes
            TPO (1:6,tcount_o) = Newdata(i,1:6);
            TPO (7:9,tcount_o) = Newdata(i,8:10);
            tcount_o = tcount_o + 1;
        else
            % no
            TPX (1:6,tcount_x) = Newdata(i,1:6);
            TPX (7:9,tcount_x) = Newdata(i,8:10);
            tcount_x = tcount_x + 1;
        end        
        % normal data list
        VD (1:6,1,1,valdata) = Newdata(i,1:6);
        VD (7:9,1,1,valdata) = Newdata(i,8:10);
        VL (valdata,1) = Newdata(i,7);
        valdata = valdata + 1;
    else
        % for training
            % participate in T2?
        if Newdata(i,7)
            % yes
            TPO (1:6,tcount_o) = Newdata(i,1:6);
            TPO (7:9,tcount_o) = Newdata(i,8:10);
            tcount_o = tcount_o + 1;
        else
            % no
            TPX (1:6,tcount_x) = Newdata(i,1:6);
            TPX (7:9,tcount_x) = Newdata(i,8:10);
            tcount_x = tcount_x + 1;
        end   
        TD (1:6,1,1,traindata) = Newdata(i,1:6);
        TD (7:9,1,1,traindata) = Newdata(i,8:10);
        TL (traindata,1) = Newdata(i,7);
        traindata = traindata+1;
    end
end
VData{1,1} = TPO;
VData{2,1} = TPX;
TData{1,1} = TPO;
TData{2,1} = TPX;

% VL = categorical(VL);
TL = categorical(TL);


% define layers
layers = [
    imageInputLayer([9 1 1])

    convolution2dLayer(2,16,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(1,'Stride',2)

    convolution2dLayer(2,64,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(1,'Stride',2)

    convolution2dLayer(2,256,'Padding','same')
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

% set options
    Current = 'adam'; % select option -> will be file name rate = .001
%     Current = 'sgdm'; % select option -> will be file name rate = .01
% Current = 'rmsprop';
options = trainingOptions(Current, ...
    'Shuffle','once', ...
    'Verbose',false, ...
    'MiniBatchSize',round(leng/5),...
    'ExecutionEnvironment','auto', ...
    'Plots','training-progress');
    % 'Plots','training-progress',...
    %'InitialLearnRate',0.001, ...
[net,info] = trainNetwork(TD,TL,layers,options);

% fprintf("V %f\n",info.ValidationAccuracy(size(info.ValidationAccuracy,2)));
fprintf("T %f\n",info.TrainingAccuracy(size(info.TrainingAccuracy,2)));

% generate random data (10000 entities)
random_count = 10000;
random_data = zeros(9,random_count);

% Gender = 1 or 2
random_data(1,:) = round(rand(1,random_count))+1;
% attn = 1 or 0, unattn = 1- attn
random_data(2,:) = round(rand(1,random_count));
random_data(3,:) = 1- random_data(2,:);
% Int = 1.66667 to 7
random_data(5,:) = round((rand(1,random_count)* (7-1.666667) + 1.666667)*3)/3;
% Vols = 1 to 3
random_data(6,:) = round( (rand(1,random_count) * 2 + 1)*4)/4;
%Touch
random_data(7,:) = round(rand(1,random_count)*3+1);
%Excel
random_data(8,:) = round(rand(1,random_count)*3+1);
%Diff
random_data(9,:) = round(rand(1,random_count)*3+1);

% reform
random_datad = zeros(9,1,1,random_count);
random_datad(:,1,1,:) = random_data(:,:);

% predict
TPLabels = predict(net,random_datad);
% result
predicted = TPLabels(:,1) >= TPLabels(:,2);

% ratio data by gender
ratio_gender = zeros(2,2);
for i=1:random_count
    if predicted(i)
        % participated
        ratio_gender(random_data(1,i),1) = ratio_gender(random_data(1,i),1) + 1;
    else
        % not
        ratio_gender(random_data(1,i),2) = ratio_gender(random_data(1,i),2) + 1;
    end
end

% ratio data by attn
ratio_attn=zeros(2,2);
for i=1:random_count
    if predicted(i)
        % participated
        ratio_attn(random_data(2,i)+1,1) = ratio_attn(random_data(2,i)+1,1) + 1;
    else
        % not
        ratio_attn(random_data(2,i)+1,2) = ratio_attn(random_data(2,i)+1,2) + 1;
    end
end

% ratio data by Touch
ratio_touch=zeros(4,2);
for i=1:random_count
    if predicted(i)
        % participated
        ratio_touch(random_data(7,i),1) = ratio_touch(random_data(7,i),1) + 1;
    else
        % not
        ratio_touch(random_data(7,i),2) = ratio_touch(random_data(7,i),2) + 1;
    end
end

% ratio data by Excel
ratio_excel=zeros(4,2);
for i=1:random_count
    if predicted(i)
        % participated
        ratio_excel(random_data(8,i),1) = ratio_excel(random_data(8,i),1) + 1;
    else
        % not
        ratio_excel(random_data(8,i),2) = ratio_excel(random_data(8,i),2) + 1;
    end
end


% ratio data by Diff
ratio_diff=zeros(4,2);
for i=1:random_count
    if predicted(i)
        % participated
        ratio_diff(random_data(9,i),1) = ratio_diff(random_data(9,i),1) + 1;
    else
        % not
        ratio_diff(random_data(9,i),2) = ratio_diff(random_data(9,i),2) + 1;
    end
end

% ratio by Incremental theory
% how many numbers? 17.
UNIQUE = unique(random_data(5,:));
ratio_int = zeros(size(UNIQUE,2),2);
for i=1:random_count
    % find index
    [c index] = min(abs(UNIQUE-random_data(5,i)));
    if predicted(i)
        
        
        % participated
        ratio_int(index,1) = ratio_int(index,1) + 1;
    else
        % not
        ratio_int(index,2) = ratio_int(index,2) + 1;
    end
end

% ratio by vols
% how many numbers? 17.
UNIQUE1 = unique(random_data(6,:));
ratio_vols = zeros(size(UNIQUE1,2),2);
for i=1:random_count
    % find index
    [c index] = min(abs(UNIQUE1-random_data(6,i)));
    if predicted(i)
        
        
        % participated
        ratio_vols(index,1) = ratio_vols(index,1) + 1;
    else
        % not
        ratio_vols(index,2) = ratio_vols(index,2) + 1;
    end
end