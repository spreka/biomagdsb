function [y, p, t_tr, t_pr] = sacCrossValidate(modelStr, data, folds, quiet)
% [y, p] = sacCrossValidate(modelStr, data, folds)
% Applies a classifier specified by 'modelStr' to 'data' using k-folds
% cross-validation where the number of folds is 'folds'.
%
% Input:
%   modelStr:       TODO.
%   data:           XX
%   folds:          The number of folds to split the data.
%   quiet:          1 = Quiet mode
%
% Output:
%   y:              The predicted class labels.
%   p:              Class-wise prediction probabilities.
%   t_tr:           Time elapsed during training
%   t_pr:           Time elapsed during prediction
%
% See also: sacKFolds, sacEvaluate, sacTrain, sacPredict

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




% TODO: remove this! it is here for job submitting!
%sacInitJava()


if ~exist('quiet', 'var')
    quiet = 0;
end

if ~quiet
    tic;
end

% k-fold random seed
seed = 0;

% split the data in k folds
data = sacKFolds(data, folds, seed);

N = numel(data.labels);
M = numel(data.classNames);
y = zeros(N,1);
p = zeros(N,M);
t_tr = zeros(folds, 1);
t_pr = zeros(folds, 1);

for k = 1:folds

    % train
    if ~quiet
        disp(['=== fold [' num2str(k) '/' num2str(folds) '] ===']);
        disp('   training');
    end
    
    t1 = tic;
    mdl = sacTrain(modelStr, data, k);
    t_tr(k) = toc(t1);
    
    % predict
    if ~quiet
        disp('   predicting');
    end
    
    t2 = tic;
    [yk, pk] = sacPredict(mdl, data, k);
    t_pr(k) = toc(t2);
    
    inds = data.testFolds{k};
    
    y(inds) = yk;
    p(inds,:) = pk;
     
    if ~quiet
        disp(' ');
    end
end

if ~quiet
    toc;
end
