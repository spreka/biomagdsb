function data = convertACC2SALT(CommonHandles)
% AUTHOR:	Peter Horvath
% DATE: 	April 22, 2016
% NAME: 	convertACC2SALT
% 
% It converts the training set to be used with SALT.
%
% INPUT:
%   CommonHandles   List of parameters of ACC.
%
% OUTPUT:
%   data            Structure to run SALT.
%
% COPYRIGHT
% Advanced Cell Classifier (ACC) Toolbox. All rights reserved.
% Copyright (C) 2016 Peter Horvath,
% Synthetic and System Biology Unit, Hungarian Academia of Sciences,
% Biological Research Center, Szeged, Hungary; Institute for Molecular
% Medicine Finland, University of Helsinki, Helsinki, Finland.

% prepare data for SALT
data.name = CommonHandles.ProjectName;
% number of instances
N = length(CommonHandles.TrainingSet.Features);
% number of features
FS = length(CommonHandles.TrainingSet.Features{1});
% number of classes
CS = length(CommonHandles.TrainingSet.Class{1});
for i=1:FS
    data.featureName{i} = ['feature' num2str(i)];
    data.featureTypes{i} = 'NUMERIC';
end;
for i=1:CS
    data.classNames{i} =  ['class' num2str(i)];
end;
data.instances = cell2mat(CommonHandles.TrainingSet.Features);

for i=1:N
    [maxv, data.labels(i)] = max(CommonHandles.TrainingSet.Class{i}(:));
end;