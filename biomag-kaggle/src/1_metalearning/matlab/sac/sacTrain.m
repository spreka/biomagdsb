function model = sacTrain(cin, data, fold)
% MODEL = sacTrain(classifier_string, data, fold)
% MODEL = sacTrain(classifier_struct, data, fold)
% Trains a specified classifier(s) using the supplied data. 
%
% INPUT:
%  classifier_string: a string the specifies the type of classifier to
%                     train followed by any classifier-specific options
%                     (see sacOptions for details)
%  classifier_struct: a structure specifying a list or heirarchy of
%                     classifiers to train, as well as the type of ensemble
%                     (voting, bagging, etc.)
%                     TODO: include further detail about ensemble meths
%
% OUTPUT:
%  model:       a structure containing the trained classifier model   
%  
% See also: sacPredict, sacNormalizeData, sacOptions, sacListClassifiers

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

if ischar(cin)
    % break the input into a classifier string and parameter string
    cid = strtok(cin);
    paramstr = strtrim(strrep(cin, cid, ''));

    % create a function handle to the appropriate function
    funstr = ['sacTrain_' cid ];
    fh = str2func(funstr);

    % check that we have a valid classifier function
    if ~exist(funstr, 'file')
        error(['Unrecognized classifier function: ' funstr '. Check that a corresponding file exists in sacROOT/clasifiers/']);
    end
    
    % check if we use the entire data set or a cross-validation fold
    if isfield(data, 'trainFolds') && exist('fold', 'var')
        data.instances = data.instances(data.trainFolds{fold},:);
        data.labels = data.labels(data.trainFolds{fold},:);
    end
    
%     size(data.instances)
%     size(data.labels)

    % call the training function and return the trained classifier model
    model = fh(data, paramstr);

elseif isstruct(cin)
    % break the input into a classifier string and parameter string
    cid = cin.type;
    %paramstr = strtrim(cin.Options);
    
    % create a function handle to the appropriate function
    funstr = ['sacTrain_' cid ];
    fh = str2func(funstr);
    
    % check that we have a valid classifier function
    if ~exist(funstr, 'file')
        error(['Unrecognized classifier function: ' funstr '. Check that a corresponding file exists in sacROOT/clasifiers/']);
    end

	% check if we use the entire data set or a cross-validation fold
    if isfield(data, 'trainFolds') && exist('fold', 'var')
        data.instances = data.instances(data.trainFolds{fold},:);
        data.labels = data.labels(:,data.trainFolds{fold});
    end
    
    % call the training function and return the trained classifier model
    model = fh(data, cin);
    
end