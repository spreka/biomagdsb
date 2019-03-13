function [y, p] = sacPredict_SVM_Libsvm(classifier, data)
% [p,y] = sacLibsvmPredict(classifier, data)
% Applies an SVM classifier using the LIBSVM library to data. 
%
% INPUT:
%  classifier:  a data structure containing a sac classifier provided
%               by sacTrain
%  data:        a sac data structure containing the training data
%
% OUTPUT:
%  y:           vector containing predicted classes for each data instance    
%  p:           matrix containing probability distribution for all classes
%  
% See also: sacLibsvmTrain, sacTrain, sacNormalizeData

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


% TODO: linear kernel does not support probability predictions!

%if strcmp(
cmd = '-b 1';
%cmd = '';

[y, acc, p] = svmpredict(data.labels, data.instances, classifier.model{1}, cmd); %#ok<ASGLU>

%keyboard;

% the order of the class prediction might need to be sorted
[s inds] = sort(classifier.model{1}.Label); %#ok<ASGLU>
p = p(:,inds);

