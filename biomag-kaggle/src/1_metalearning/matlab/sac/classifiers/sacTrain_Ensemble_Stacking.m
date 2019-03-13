function model = sacTrain_Ensemble_Stacking(data, cls)
% classifier = sacTrain_Ensemble_Stacking(data, classifierStruct)
% Trains a meta classifier using voting scheme. 
%
% INPUT:
%  data:            a sac data structure containing the training data
%  paramStruct:     a string containing options for training. Options
%                   include
%                   -norm Data normalization type
%                    
%                   Default options: 
% 
% OUTPUT:
%  classifier       a data structure containing a sac classifier
%
% See also: sacPredict_Ensemble_Stacking, sacTrain, sacNormalizeData

% From the Suggest a Classifier Library (SAC), a Matlab Toolbox for cell
% classification in high content screening. http://www.cellclassifier.org/
% Copyright © 2011 Kevin Smith and Peter Horvath, Light Microscopy Centre 
% (LMC), Swiss Federal Institute of Technology Zurich (ETHZ), Switzerland. 
% All rights reserved.
%
% This program is free software; you can redistribute it and/or modify it 
% under the terms of the GNU General Public License version 2 (or higher) 
% as published by the Free Software Foundation. This program is 
% distributed WITHOUT ANY WARRANTY; without even the implied warranty of 
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
% General Public License for more details.



% handle parameter string
if ~isfield(cls, 'Options')
    parameters = '';
else
    if (isempty(cls.Options)) || strcmp(cls.Options, '')
        parameters = '';
    else
        parameters = cls.Options;
    end
end
    



% handle data normalization
[parameters normFlag] = sacNormHandler(parameters);
if normFlag
    [data dataRange] = sacNormalizeData(data, normFlag);
end


num_classifiers = length(cls.SubClassifiers);
num_classes = numel(data.classNames);
N = numel(data.labels);


%% Train the sub-classifiers
if isfield(cls, 'SubClassifiers')
    
    if normFlag
        model.dataRanges = dataRange;
    else
        model.dataRanges = [];
    end
    model.model = {};
    model.normalizeData = normFlag;
    model.type = 'EnsembleVote';
    model.options = parameters;
    
    for i = 1:num_classifiers
        
        ci = cls.SubClassifiers(i);
        
        % no not normalize the data for sub-classifiers
        normstr = regexp(parameters, '-norm\s*\d*', 'match');
        if isempty(normstr)
            ci.Options = strtrim(['-norm 0' ci.Options]);
        end
        
        
        % break the input into a cls string and parameter string
        cid = ci.ClassifierType;
       
        disp(['     training sub-classifier "' cid '"']);
    
        % create a function handle to the appropriate function
        funstr = ['sacTrain_' cid ];
        fh = str2func(funstr);
        
        % check that we have a valid classifier function
        if ~exist(funstr, 'file')
            error(['Unrecognized classifier function: ' funstr '. Check that a corresponding file exists in sacROOT/clasifiers/']);
        end
        
        % train subclassifier i
        model.SubClassifiers(i) = fh(data, ci);
        
    end
    
else
    error('No classifiers were specified to be combined.')
end


%keyboard;

%% get sub-classifier predictions as new training data
% TODO: make use of validation data

%load d2.mat;
%data = d2;
%data  = sacNormalizeData(data, normFlag);


disp('     making meta-classifier predictions');
meta_data = zeros(N,num_classifiers*num_classes);

for i = 1:num_classifiers
    
    ci = model.SubClassifiers(i);
    
    %disp(['...meta-classifier predictions for "' ci.type '"']);
    
    % create a function handle to the appropriate function
    funstr = ['sacPredict_' ci.type ];
    fh = str2func(funstr);


    % check if the function exists
    if ~exist(funstr, 'file')
        error(['Unrecognized classifier prediction function: ' funstr '. Check that a corresponding file exists in sacROOT/clasifiers/']);
    end
    
    inds = [1:num_classes] + (i-1)*num_classes; %#ok<NBRAK>
    
    [~, meta_data(:,inds)] = fh(ci,data);
end

% replace the original data with meta-data
data.instance = meta_data;





%% train the stacked classifier on the meta-data

% TODO: normalization of the meta-data? (problably not)



% break the input into a cls string and parameter string, the meta-classifier 
% is specified as the first token in the options
[cid remain] = strtok(cls.Options);
cls.Options = remain;

disp(['   training stacked classifier "' cid '"']);

% no not normalize the data
normstr = regexp(cls.Options, '-norm\s*\d*', 'match');
if isempty(normstr)
    cls.Options = strtrim(['-norm 0' cls.Options]);
end

% create a function handle to the appropriate function
funstr = ['sacTrain_' cid ];
fh = str2func(funstr);

% check that we have a valid classifier function
if ~exist(funstr, 'file')
    error(['Unrecognized classifier function: ' funstr '. Check that a corresponding file exists in sacROOT/clasifiers/']);
end

% train 
meta_model = fh(data, cls);
        
model.model = meta_model.model;
model.options = [cid ' ' remain];
%model.type = 'EnsembleStacking';
str = mfilename();
model.type = str(strfind(str,'_')+1:end);


model = orderfields(model);



