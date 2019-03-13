function classifier = sacTrain_LogitBoost_Weka(data, parameters)
% classifier = sacTrain_LogitBoost_Weka(data, parameters)
% Trains a LogitBoost classifier using the WEKA library. 
%
% INPUT:
%  data:            a sac data structure containing the training data
%  parameters:      a string containing options for training. Options
%                   include
%                   -norm Data normalization type
%                   -H Hueristic stop. If > 0, the heuristic for greedy 
%                      stopping while cross-validating the number of 
%                      LogitBoost iterations is enabled. This means 
%                      LogitBoost is stopped if no new error minimum has 
%                      been reached in the last heuristicStop iterations.
%                   -M Max boosting iterations.
%                   -I Sets fixed number of boosting iterations.
%                   -W Sets beta value for weight trimming.
%
% OUTPUT:
%  classifier       a data structure containing a sac classifier
%
% See also: sacPredict_LogitBoost_Weka, sacTrain, sacNormalizeData

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


classifierWekaType = weka.classifiers.functions.SimpleLogistic;

filestr = mfilename();
classifier = sacWekaTrain(classifierWekaType, data, parameters, filestr);
