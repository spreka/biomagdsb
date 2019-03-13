function [y,p] = sacPredict(classifier, data, fold)
%function [y,p] = sacPredict(classifier, data, fold)
% Applies a LogitBoost classifier using the WEKA library to data. 
%
% INPUT:
%  classifier:  a data structure containing a sac classifier provided
%               by sacTrain
%  data:        a sac data structure containing the training data
%
% OUTPUT:
%  p:           probability distribution for each class
%  y:           vector containing predicted classes for each data instance    
%  
% See also: sacLogitBoostTrain, sacTrain, sacNormalizeData

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


% TODO: figure out if it is a single classifier or ensemble



% create a function handle to the appropriate function
funstr = ['sacPredict_' classifier.type ];
fh = str2func(funstr);

% normalize the data (if necessary)
if classifier.normalizeData
    data = sacNormalizeData(data, classifier.normalizeData, classifier.dataRanges);
end

% check if the function exists
if ~exist(funstr, 'file')
    error(['Unrecognized classifier prediction function: ' funstr '. Check that a corresponding file exists in sacROOT/clasifiers/']);
end

% check if we use the entire data set or a cross-validation fold
if isfield(data, 'testFolds') && exist('fold', 'var')
    data.instances = data.instances(data.testFolds{fold},:);
    data.labels = data.labels(data.testFolds{fold});
end

% call the prediction function
[y,p] = fh(classifier, data);


