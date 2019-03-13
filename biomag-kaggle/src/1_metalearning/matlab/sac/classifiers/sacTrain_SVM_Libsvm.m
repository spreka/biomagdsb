function classifier = sacTrain_SVM_Libsvm(data, parameters)
% classifier = sacLibsvmTrain(data, parameters)
% Trains a SVM classifier using the LIBSVM library. 
%
% INPUT:
%  data:            a sac data structure containing the training data
%  parameters:      a string containing options for training. Options
%                   include
%                   -norm Data normalization type
%                   -v n: n-fold cross validation
%                   -t kernel type (2 = radial basis function)
%                   -q quiet mode
%                   -c cost
%                   -g gamma in the kernel function
%                   -m cache size
%                   -b probability_estimates: whether to train a SVC or SVR 
%                      model (0=SVC)
% 
%                   Default options: -t 2 -q -m 500 -b 1
%                   Further details at http://www.csie.ntu.edu.tw/~cjlin/libsvm/
% 
% OUTPUT:
%  classifier       a data structure containing a sac classifier
%
% See also: sacPredict_SVM_Libsvm, sacTrain, sacNormalizeData

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


% TODO: make sure LIBSVM is properly installed
% TODO: make sure svmtrain and svmpredict point to the libsvm libraries

% TODO: some basic error checking of the parameters string 



% get the name of this classifier
str = mfilename();
classifier.type = str(strfind(str,'_')+1:end);


% provide a default parameter string if none is provided
if isempty(parameters)
    [parameters OPT] = sacGetDefaultOptions(classifier.type); %#ok<ASGLU>
    parameters = parameters{1};
elseif isstruct(parameters)
    % support possible passing of a structure instead of string
    parameters = parameters.Options;
end

% handle data normalization
[parameters normFlag] = sacNormHandler(parameters);
if normFlag
    [data dataRange] = sacNormalizeData(data, normFlag);
end


% train the svm
%cmd = [parameters ' -b 1'];
cmd = parameters;
classifier.model = {svmtrain(data.labels', data.instances, cmd)};

% set details of the classifier struct
classifier.normalizeData = normFlag;
if normFlag
    classifier.dataRanges = dataRange;
end
classifier.options = parameters;
classifier.SubClassifiers = [];
classifier = orderfields(classifier);





              