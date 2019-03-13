function classifier = sacTrain_NearestNeighbor_Weka(data, parameters)
% classifier = sacTrain_NearestNeighbor_Weka(data, parameters)
% Trains a nearest neighbor classifier using the WEKA library. 
%
% INPUT:
%  data:            a sac data structure containing the training data
%  parameters:      a string containing options for training. Options
%                   include
%                   -norm Data normalization type
%
% OUTPUT:
%  classifier       a data structure containing a sac classifier
%
% See also: sacPredict_NearestNeighbor_Weka, sacTrain, sacNormalizeData

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

%theres no more weka.classifier.lazy.IB1 in newer version of weka
classifierWekaType = weka.classifiers.lazy.IBk;
filestr = mfilename();
classifier = sacWekaTrain(classifierWekaType, data, parameters, filestr);