function classifier = sacWekaTrain(classifierWekaType, data, parameters, str)
% classifier = sacWekaTrain(data, parameters)
% Trains a Weka classifier 
%
% INPUT:
%  data:            a sac data structure containing the training data
%  parameters:      a string containing options for training. Options
%                   include
%                   -norm Data normalization type
%                   -E estimator: Estimator algorithm for finding the 
%                      conditional probability tables of the Bayes Network
%                      (BayesNetEstimator, SimpleEstimator, ...)
%                   -S searchAlgorithm: method used for searching network 
%                      structures (K2, HillClimber, ...)
%
% OUTPUT:
%  classifier       a data structure containing a sac classifier
%
% See also: sacPredict_BayesNet_Weka, sacTrain, sacNormalizeData

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


% get the name of this classifier
classifier.type = str(strfind(str,'_')+1:end);

% provide a default parameter string if none is provided
if isempty(parameters)
    [paramcell OPT] = sacGetDefaultOptions(classifier.type); %#ok<NASGU>
    parameters = paramcell{1}; clear OPT;
elseif isstruct(parameters)
    % support possible passing of a structure instead of string
    parameters = parameters.options;
end

% handle data normalization
[parameters normFlag] = sacNormHandler(parameters);
if normFlag
    [data dataRange] = sacNormalizeData(data, normFlag);
end

% pass the options string to the WEKA classifier
classifierWekaType.setOptions(weka.core.Utils.splitOptions(parameters));

% define the number of features, classes, and instances
numfeats = size(data.instances,2);
numclass = numel(data.classNames); 
N = size(data.instances, 1);

% enumerate the attributes (features)
attributes = weka.core.FastVector(numfeats+1);
for i=1:numfeats
    attributes.addElement(weka.core.Attribute(['feature' num2str(i)]));
end

% enumerate the classes
classvalues = weka.core.FastVector(numclass);    
for i=1:numclass
    classvalues.addElement(['class' num2str(i)]);
end
attributes.addElement(weka.core.Attribute('Class', classvalues));

% create WEKA data class
trainingdata = weka.core.Instances('training_data', attributes, N);
trainingdata.setClassIndex(trainingdata.numAttributes() - 1);

% fill trainingdata with instances containing values from 'data'
w = 1;
for i = 1:N
    inst = weka.core.DenseInstance(w, [data.instances(i,:) 0]);
    inst.setDataset(trainingdata);
  	classLabel = data.labels(i);
    inst.setClassValue(['class' num2str(classLabel)]);
    trainingdata.add(inst);
end

% build the classifier
classifierWekaType.buildClassifier(trainingdata);     


% store the classifer model and other useful information in a structure
classifier.model  = {classifierWekaType};
classifier.options = parameters;
classifier.normalizeData = normFlag;
if normFlag
    classifier.dataRanges = dataRange;
end
classifier.SubClassifiers = [];
classifier = orderfields(classifier);


