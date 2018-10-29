clear all;

% load test data
M = csvread("test1.csv");
% delete id
Newdata = M(:,2:11);
% set aside test data
Fulldata = Newdata;
%Newdata = Newdata(1:size(Newdata,1)-30,:);
leng = size(Newdata,1);

% test 1000 times
for trials =1:1000

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
    vcount_o = 1;
    vcount_x = 1;
    tcount_o = 1;
    tcount_x = 1;
    for i=1:leng
        % for validation
        if (r(i) <= leng/3)
        % participate in T2?
            if Newdata(i,7)
                % yes
                VPO (1:6,vcount_o) = Newdata(i,1:6);
                VPO (7:9,vcount_o) = Newdata(i,8:10);
                vcount_o = vcount_o + 1;
            else
                % no
                VPX (1:6,vcount_x) = Newdata(i,1:6);
                VPX (7:9,vcount_x) = Newdata(i,8:10);
                vcount_x = vcount_x + 1;
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
    VData{1,1} = VPO;
    VData{2,1} = VPX;
    TData{1,1} = TPO;
    TData{2,1} = TPX;

    VL = categorical(VL);
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
%     Current = 'adam'; % select option -> will be file name rate = .001
%     Current = 'sgdm'; % select option -> will be file name rate = .01
    Current = 'rmsprop';
    options = trainingOptions(Current, ...
        'MaxEpochs',200, ...
        'Shuffle','once', ...
        'ValidationData',{VD,VL}, ...
        'ValidationFrequency',5, ...
        'ValidationPatience',5,...
        'Verbose',false, ...
        'MiniBatchSize',round(leng/5),...
        'ExecutionEnvironment','auto');
        % 'Plots','training-progress',...
        %'InitialLearnRate',0.001, ...
    [net,info] = trainNetwork(TD,TL,layers,options);

    fprintf("%d %f\n",  trials,info.ValidationAccuracy(size(info.ValidationAccuracy,2)));
    fprintf("%d %f\n",  trials,info.TrainingAccuracy(size(info.ValidationAccuracy,2)));
    
    Acc_Val(trials) = info.ValidationAccuracy(size(info.ValidationAccuracy,2));
    Acc_Trn(trials) = info.TrainingAccuracy(size(info.ValidationAccuracy,2));
end

% save results
save (sprintf('%s_result.mat',Current),'Acc_Val','Acc_Trn');