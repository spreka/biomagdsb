function [y, p] = sacWekaPredictProb(classifier, data)
% [y, p] = sacWekaPredictProb(classifier, data)
% Prediction for classifiers using the WEKA library for classifiers that
% support the "distributionForInstance" method.
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
end;
attributes.addElement(weka.core.Attribute('Class', classvalues));

% create the WEKA instances class to hold the data
d = weka.core.Instances('training_data', attributes, N);
d.setClassIndex(d.numAttributes() - 1);

% fill the data set with instances
w = 1;
for i = 1:N
    inst = weka.core.DenseInstance(w, [data.instances(i,:) 0]);
    inst.setDataset(d);
  	classLabel = data.labels(i);
    inst.setClassValue(['class' num2str(classLabel)]);   
    d.add(inst);
end

% fill the output vector classification results and probabilities
y = zeros(N,1);
p = zeros(N,numclass);
mdl = classifier.model{1};
for i=0:N-1    
    %y(i+1) = d.instance(i).classValue+1; 
    p(i+1, :) = mdl.distributionForInstance(d.instance(i));
    [dummy,y(i+1)] = max(p(i+1, :)); %#ok<ASGLU>
    clear dummy;
end




% % get probability distributions
% for i=0:1:N-1       
%     props(i+1, :) = classifier.model.distributionForInstance(d.instance(i));
%     [~,out(i+1)] = max(props(i+1, :));
% end