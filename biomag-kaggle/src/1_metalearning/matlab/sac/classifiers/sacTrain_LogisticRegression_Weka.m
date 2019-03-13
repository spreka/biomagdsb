function classifier = sacTrain_LogisticRegression_Weka(data, parameters)
% classifier = sacTrain_Logistic_Weka(data, parameters)
% Trains a logistic regression classifier using the WEKA library. 
%
% INPUT:
%  data:            a sac data structure containing the training data
%  parameters:      a string containing options for training. Options
%                   include
%                   -norm Data normalization type
%                   -M maxits: maximum number of iterations
%                   -R ridgeval: the Ridge value in the log-likelihood
%
% OUTPUT:
%  classifier       a data structure containing a sac classifier
%
% See also: sacPredict_Logistic_Weka, sacTrain, sacNormalizeData

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


classifierWekaType = weka.classifiers.functions.Logistic;

filestr = mfilename();
classifier = sacWekaTrain(classifierWekaType, data, parameters, filestr);