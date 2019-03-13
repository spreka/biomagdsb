function [ out, props, CommonHandles ] = predictClassifier( features , CommonHandles)
% AUTHOR:   Peter Horvath, Abel Szkalisity
% DATE:     April 22, 2016
% NAME:     predictClassifier
%
% To run prediction on the input features. For the successful run it needs
% to have a trained classifier stored in CommonHandles, to have this the
% trainClassifier function should be called beforehand.
%
% INPUT:
%   features     	The input features on which the prediction will run.
%                   Format: array, the rows refers to observations.
%
% OUTPUT:
%   out             a 1D array with the predicted classes. It has as many
%                   rows as the features.
%   props           Class probabilities for all possible classes. A matrix
%                   with the same number of rows as features and it has as
%                   many column as many possible classes we have. Usually
%                   the greatest class probability also indicates the
%                   predictied class label (out variable).
%
% COPYRIGHT
% Advanced Cell Classifier (ACC) Toolbox. All rights reserved.
% Copyright (C) 2016 Peter Horvath,
% Synthetic and System Biology Unit, Hungarian Academia of Sciences,
% Biological Research Center, Szeged, Hungary; Institute for Molecular
% Medicine Finland, University of Helsinki, Helsinki, Finland.

%global CommonHandles

oneRow = features(1,:); %#ok<NASGU> Used below in whos
w = whos('oneRow');
bigDataLimit = round((128*2^20)/w.bytes);

featureSize = size(features,1);

%screensize = get(0,'Screensize');

%if we can manage the predict with one simple call
if featureSize<=bigDataLimit
        
    %infoD = dialog('Name','Predict classifier','Position',[screensize(3)/2-150, screensize(4)/2-40,300,80]);
    %uicontrol('Parent',infoD,'Style','text','Units','Pixels','Position',[30 20 240 40],'String','Classifing cells based on your training... (Prediction) This will not take long.');    
    
    if CommonHandles.SALT.initialized
        CommonHandles.SALT.trainingData.instances = features;
        CommonHandles.SALT.trainingData.labels = ones(size(features, 1),1);
        if strcmp(CommonHandles.SelectedClassifier,'OneClassClassifier')
            [out, ~ ,props] = svmpredict(CommonHandles.SALT.trainingData.labels, CommonHandles.SALT.trainingData.instances, CommonHandles.SALT.model);        
            out(out<0) = 2;
        else
            [out,props] = sacPredict(CommonHandles.SALT.model, CommonHandles.SALT.trainingData);
        end
    else
        errordlg('Please initalize SALT/sac!');
    end
    
    %if ishandle(infoD)
    %    close(infoD);
    %end   
    
%otherwise do it in parallel
elseif CommonHandles.SALT.initialized        
    %save to local variables for parfor    
    globalTrainingData = CommonHandles.SALT.trainingData;
    globalModel = CommonHandles.SALT.model;
    globalSelectedClassifier = CommonHandles.SelectedClassifier;
    %slice features to batches
    nofSlices = ceil(featureSize / bigDataLimit);
    %init IO
    slices = cell(1,nofSlices);
    outCell = cell(nofSlices,1);
    propCell = cell(nofSlices,1);
    for i=1:nofSlices
        startOfBatch = (i-1)*bigDataLimit+1;
        endOfBatch = min(i*bigDataLimit,featureSize);
        slices{i} = features(startOfBatch:endOfBatch,:);
    end
    %{
    % Parallel computing is REMOVED, because it seems that sacPredict can't
    operate in parallel
    if 0 %checkToolboxByName('Parallel Computing Toolbox')
        
        infoD = dialog('Name','Predict classifier','Position',[screensize(3)/2-150, screensize(4)/2-30,300,80]);
        uicontrol('Parent',infoD,'Style','text','Units','Pixels','Position',[30 20 240 60],'String','Classifing cells based on your training... (Prediction) We use parallel computing toolbox.');
        
        parfor i=1:nofSlices
            localTrainingData = globalTrainingData;
            localTrainingData.instances = slices{i};
            localTrainingData.labels = ones(size(slices{i},1),1);
            if strcmp(globalSelectedClassifier,'OneClassClassifier')
                [outCell{i}, ~ ,propCell{i}] = svmpredict(localTrainingData.labels, localTrainingData.instances, globalModel);
                outCell{i}(outCell{i}<0) = 2;
            else
                [outCell{i},propCell{i}] = sacPredict(globalModel, localTrainingData);
            end
        end
        
        if ishandle(infoD)
            close(infoD);
        end
        
    %else
    %}
        
        %predictWaitBarHandle = waitbar(0, 'Classifing cells based on your training... (Prediction) Go and have a coffee if the line below goes very slow.');
        
        for i=1:nofSlices
            localTrainingData = globalTrainingData;
            localTrainingData.instances = slices{i};
            localTrainingData.labels = ones(size(slices{i},1),1);
            if strcmp(globalSelectedClassifier,'OneClassClassifier')
                [outCell{i}, ~ ,propCell{i}] = svmpredict(localTrainingData.labels, localTrainingData.instances, globalModel);
                outCell{i}(outCell{i}<0) = 2;
            else
                [outCell{i},propCell{i}] = sacPredict(globalModel, localTrainingData);
            end
            
        %   donePercent = i/nofSlices;
        %    if ishandle(predictWaitBarHandle)
        %        waitbar(donePercent,predictWaitBarHandle,sprintf('Classifing cells based on your training... (Prediction) Go and have a coffee if the line below goes very slow. %d%% done',int16(donePercent*100)));
        %    else
        %        waitbar(donePercent,sprintf('Classifing cells based on your training... (Prediction) Go and have a coffee if the line below goes very slow. %d%% done',int16(donePercent*100)));
        %    end
        end
        
        %if ishandle(predictWaitBarHandle)
        %    close(predictWaitBarHandle);
        %end
            
    %end
    out = cell2mat(outCell);
    props = cell2mat(propCell);
else
    errordlg('Please initalize SALT/sac!');
end