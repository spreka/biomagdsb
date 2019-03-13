function [y,p] = sacPredict_Ensemble_Stacking(classifier, data)
% [y, p] = sacPredict_Ensemble_Stacking(classifier, data)
% Applies an voting ensamble of classifiers to do prediction on the data. 
%
% INPUT:
%  classifier:  a data structure containing sac classifier(s) provided
%               by sacTrain
%  data:        a sac data structure containing the training data
%
% OUTPUT:
%  y:           vector containing predicted classes for each data instance    
%  p:           matrix containing probability distribution for all classes
%  
% See also: sacPredict_Ensemble_Stacking, sacPredict, sacNormalizeData

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

num_classifiers = length(classifier.SubClassifiers);
num_classes = numel(data.classNames);
N = numel(data.labels);



%% get sub-classifier predictions as new training data
meta_data = zeros(N,num_classifiers*num_classes);

for i = 1:num_classifiers
    
    ci = classifier.SubClassifiers(i);
    
    disp(['     meta-classifier predictions for "' ci.type '"']);
    
    % create a function handle to the appropriate function
    funstr = ['sacPredict_' ci.type ];
    fh = str2func(funstr);


    % check if the function exists
    if ~exist(funstr, 'file')
        error(['Unrecognized classifier prediction function: ' funstr '. Check that a corresponding file exists in sacROOT/clasifiers/']);
    end
    
    inds = [1:num_classes] + (i-1)*num_classes; %#ok<NBRAK>
    
    [trash, meta_data(:,inds)] = fh(ci,data); %#ok<*ASGLU>
end

% replace the original data with meta-data
data.instance = meta_data;


%% predict on the meta-data

[cid remain] = strtok(classifier.options);
classifier.Options = remain;


% create a function handle to the appropriate function
funstr = ['sacPredict_' cid ];
fh = str2func(funstr);

% check that we have a valid classifier function
if ~exist(funstr, 'file')
    error(['Unrecognized classifier function: ' funstr '. Check that a corresponding file exists in sacROOT/clasifiers/']);
end

disp(['   stacked predictions for "' ci.type '"']);
[y,p] = fh(classifier, data);

