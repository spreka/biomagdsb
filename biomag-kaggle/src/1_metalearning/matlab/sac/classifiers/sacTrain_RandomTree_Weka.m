function classifier = sacTrain_RandomTree_Weka(data, parameters)
% classifier = sacTrain_RandomTree_Weka(data, parameters)
% Trains a tree classifier that considers K randomly chosen attributes per 
% node using the WEKA library. 
%
% INPUT:
%  data:            a sac data structure containing the training data
%  parameters:      a string containing options for training. Options
%                   include
%                   -norm Data normalization type
%                   -K 	Number of randomly chosen attributes per node.
%                       If 0, log_2(num_total_attributes) is used.
%                   -M 	Min total weight of instances in leaf.
%                   -depth  Maximum depth of tree (0=unlimited).
%                   -N 	Amount of data used for backfitting [0,1].
% 
%                   Default options: -K 0 -M 1.0
%                   Further details at http://weka.sourceforge.net/doc/weka/classifiers/trees/RandomTree.html
% 
% OUTPUT:
%  classifier       a data structure containing a sac classifier
%
% See also: sacPredict_RandomTree_Weka, sacTrain, sacNormalizeData

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

classifierWekaType = weka.classifiers.trees.RandomTree;

filestr = mfilename();
classifier = sacWekaTrain(classifierWekaType, data, parameters, filestr);

% % support possible passing of a structure instead of string
% if isstruct(parameters)
%     parameters = parameters.Options;
% end
% 
% % handle data normalization
% [parameters normFlag] = sacNormHandler(parameters);
% if normFlag
%     [data dataRange] = sacNormalizeData(data, normFlag);
% end
% 
% % check the parameters string, perform data normalization if necessary
% if isempty(parameters)
%     % provide a default parameter string if none is provided
%     parameters = '-K 0 -M 1.0';
% end
% 
% % define the WEKA classifier type and pass its options
% classifierWekaType = weka.classifiers.trees.RandomTree;
% classifierWekaType.setOptions(weka.core.Utils.splitOptions(parameters));
% 
% % define the number of features, classes, and instances
% numfeats = size(data.instances,2);
% numclass = numel(data.classNames); 
% N = size(data.instances, 1);
% 
% % enumerate the attributes (features)
% attributes = weka.core.FastVector(numfeats+1);
% for i=1:numfeats
%     attributes.addElement(weka.core.Attribute(['feature' i]));
% end
% 
% % enumerate the classes
% classvalues = weka.core.FastVector(numclass);    
% for i=1:numclass
%     classvalues.addElement(['class' i]);
% end
% attributes.addElement(weka.core.Attribute('Class', classvalues));
% 
% % create WEKA data class
% trainingdata = weka.core.Instances('training_data', attributes, N);
% trainingdata.setClassIndex(trainingdata.numAttributes() - 1);
% 
% % fill trainingdata with instances containing values from 'data'
% w = 1;
% for i = 1:N
%     inst = weka.core.Instance(w, [data.instances(i,:) 0]);
%     inst.setDataset(trainingdata);
%   	classLabel = data.labels(i);
%     inst.setClassValue(['class' classLabel]);
%     trainingdata.add(inst);
% end
% 
% % build the classifier
% classifierWekaType.buildClassifier(trainingdata);     
% 
% % store the classifer model and other useful information in a structure
% classifier.model  = {classifierWekaType};
% %classifier.type = 'RandomTree';
% str = mfilename();
% classifier.type = str(strfind(str,'_')+1:end);
% classifier.options = parameters;
% classifier.normalizeData = normFlag;
% if normFlag
%     classifier.dataRanges = dataRange;
% end
% classifier.SubClassifiers = [];
% classifier = orderfields(classifier);