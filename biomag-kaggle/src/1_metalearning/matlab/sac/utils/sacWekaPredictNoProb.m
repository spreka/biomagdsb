function [y, p] = sacWekaPredictNoProb(classifier, data)
% [y, p] = sacWekaPredictProb(classifier, data)
% Prediction for classifiers using the WEKA library for classifiers that do
% not support the "distributionForInstance" method.
%
% INPUT:
%  classifier:  a data structure containing a sac classifier provided
%               by sacTrain
%  data:        a sac data structure containing the training data
%
% OUTPUT:
%  y:           vector containing predicted classes for each data instance 
%  p:           probability distribution for each class   
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



% the option "-soft" for certain classifiers specifies "pmax", or the
% probability assigned to the predicted class. If pmax < 1, then the
% difference is distributed among the other classes. For instance, in a
% 3-class classification problem, if the classifier predicts class 2 (but
% does not provide probabilities) and pmax = .8, then the probability
% output would be [.1 .8 .1]. The purpose of this feature is to make it
% possible to use classifiers which do not provide probabilities in certain
% ensemble methods.


% define the number of features, classes, and instances
numfeats = size(data.instances,2);  
numclass = numel(data.classNames);  
N = size(data.instances, 1);        

% enumerate the attributes (features)
attributes = weka.core.FastVector(numfeats+1);
for i=1:numfeats
    attributes.addElement(weka.core.Attribute(['feature' num2str(i)]));
end;

% enumerate the classes
classvalues = weka.core.FastVector(numclass);    
for i=1:numclass
    classvalues.addElement(['class' num2str(i)]);
end;
attributes.addElement(weka.core.Attribute('Class', classvalues));

% create the WEKA instances class to hold the data
d = weka.core.Instances('training_data', attributes, N);
d.setClassIndex(d.numAttributes() - 1);

% fill the data set with instances
for i = 1:N
    w = 1;
    inst = weka.core.Instance(w, [data.instances(i,:) 0]);

    inst.setDataset(d);
    
  	classLabel = data.labels(i);
    inst.setClassValue(['class' num2str(classLabel)]);
    
    d.add(inst);
end

% create the evaluator class
evaluator = weka.classifiers.misc.monotone.InstancesUtil;

% apply the evaluator to classify each data instance
evaluator.classifyInstances(d, classifier.model{1});

% since this WEKA classifier does not provide probabilities, we will assign
% our own (either [0,1] or some user define pmax, pmin
if isempty(strfind(classifier.options, '-soft'));
    
    % fill in probabilities with [0,1]
    y = zeros(N,1);
    p = zeros(N,numclass);
    pmax = 1;

else
    disp('here')
    % fill in soft probabilities
    t = regexpi(classifier.options, '-soft(\s*\d*\.\d*)', 'match');
    t = strrep(t{1}, '-soft', '');
    t = strtrim(t);
    pmax = str2double(t);

    y = zeros(N,1);
    pmin = (1 - pmax) / (numclass - 1);
    p = pmin * ones(N,numclass);
end


% fill the output vector classification results and probabilities
for i=0:N-1    
    y(i+1) = d.instance(i).classValue+1; 
    p(i+1, y(i+1)) = pmax;
end
    