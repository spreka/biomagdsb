function classifier = sacTrain_SVM_Weka(data, parameters)

% classifier = sacTrain_Adaboost_Weka(data, parameters)
% Trains an AdaBoost classifier using the WEKA library. 
%
% INPUT:
%  data:            a sac data structure containing the training data
%  parameters:      a string containing options for training. Options
%                   include:
%                   -norm Data normalization type
%                   -I 	ITERATIONS Number of rounds of boosting.
%                   -P 	WEIGHT THRESHOLD for pruning.
%                   -S 	SEED for random number generation.
%                   -W 	CLASSIFIER to be used (stump, tree, etc).
%
% OUTPUT:
%  classifier       a data structure containing a sac classifier
%
% See also: sacPredict_Adaboost_Weka, sacTrain, sacNormalizeData

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

classifierWekaType = weka.classifiers.functions.SMO;

filestr = mfilename();
classifier = sacWekaTrain(classifierWekaType, data, parameters, filestr);










% 
% 
% % classifier = sacTrain_SVM_Weka(data, parameters)
% % Trains a SVM classifier using the WEKA library. 
% %
% % INPUT:
% %  data:            a sac data structure containing the training data
% %  parameters:      a string containing options for training. Options
% %                   include
% %                   -norm Data normalization type
% %                   -M maxits: maximum number of iterations
% %                   -R ridgeval: the Ridge value in the log-likelihood
% %
% % OUTPUT:
% %  classifier       a data structure containing a sac classifier
% %
% % See also: sacPredict_SVM_Weka, sacTrain, sacNormalizeData
% 
% % From the Suggest a Classifier Library (SAC), a Matlab Toolbox for cell
% % classification in high content screening. http://www.cellclassifier.org/
% % Copyright © 2011 Kevin Smith and Peter Horvath, Light Microscopy Centre 
% % (LMC), Swiss Federal Institute of Technology Zurich (ETHZ), Switzerland. 
% % All rights reserved.
% %
% % This program is free software; you can redistribute it and/or modify it 
% % under the terms of the GNU General Public License version 2 (or higher) 
% % as published by the Free Software Foundation. This program is 
% % distributed WITHOUT ANY WARRANTY; without even the implied warranty of 
% % MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
% % General Public License for more details.
% 
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
%     parameters = '-R 1.0E-8 -M -1';
% end
% 
% % define the WEKA classifier type and pass its options
% classifierWekaType = weka.classifiers.functions.SMO;
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
% str = mfilename();
% classifier.type = str(strfind(str,'_')+1:end);
% classifier.options = parameters;
% classifier.normalizeData = normFlag;
% if normFlag
%     classifier.dataRanges = dataRange;
% end
% classifier.SubClassifiers = [];
% classifier = orderfields(classifier);