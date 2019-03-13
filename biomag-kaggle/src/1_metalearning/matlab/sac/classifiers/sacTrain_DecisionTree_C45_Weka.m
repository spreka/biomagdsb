function classifier = sacTrain_DecisionTree_C45_Weka(data, parameters)
% classifier = sacTrain_DecisionTree_C45_Weka(data, parameters)
% Trains a C4.5 DecisionTree classifier using the WEKA library. 
%
% INPUT:
%  data:            a sac data structure containing the training data
%  parameters:      a string containing options for training. Options
%                   include
%                   -norm Data normalization type
%                   -C 	CONF confidence factor used for pruning.
%                       Lower values incur more pruning.
%                   -M 	OBJ min number of instances per leaf.
%                   -R 	Use reduced error pruning.
%                   -N 	NUM number of folds in reduced error pruning.
% 
%                   Default options: -C 0.25 -M 2
%                   Further details at http://weka.sourceforge.net/doc/weka/classifiers/trees/J48.html
% 
% OUTPUT:
%  classifier       a data structure containing a sac classifier
%
% See also: sacPredict_DecisionTree_C45_Weka, sacTrain, sacNormalizeData

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

classifierWekaType = weka.classifiers.trees.J48;

filestr = mfilename();
classifier = sacWekaTrain(classifierWekaType, data, parameters, filestr);


