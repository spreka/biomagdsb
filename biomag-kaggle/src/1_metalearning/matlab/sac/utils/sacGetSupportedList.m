function list = sacGetSupportedList(type)


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



switch lower(type)
    
    case 'metrics'
        list = {'ACC', 'AUC', 'BEP', 'CONFUSION', 'CORRECT', 'ERR', 'FPR','FSC', 'INCORRECT', 'LFT', 'MXE', 'PR', 'PRE', 'REC', 'RMS', 'ROC','TPR'};

    case 'criteria'
        list = {'ACC', 'AUC', 'BEP', 'CONFUSION', 'CORRECT', 'ERR', 'FPR','FSC', 'INCORRECT', 'LFT', 'MXE', 'PRE', 'REC', 'RMS','TPR'};
    case 'classifiers'
        % TODO: this should be replaced by a function?
        list = {'MLP_Weka','RandomForest_Weka','LogitBoost_Weka','Adaboost_Weka', 'BayesNet_Weka', 'BoostedStump_Weka', 'DecisionTree_C45_Weka','DecisionTable_Weka','KNN_Weka','KStar_Weka','LogisticRegression_Weka','NaiveBayes_Weka','NearestNeighbor_Weka','RandomTree_Weka','SVM_Libsvm','SVM_Weka'};
    
    otherwise
        error(['Unknown type: ' type]);
end