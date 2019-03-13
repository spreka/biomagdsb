function d = sacKFolds(d, folds, seed)
% [dfold inds] = sacKFolds(data, folds, seed)
% Applies a classifier specified by 'modelStr' to 'data' using k-folds
% cross-validation where the number of folds is 'folds'.
%
% Input:
%   data:           XX
%   folds:          The number of folds to split the data.
%   seed:           [Optional] Seed for random sampling of the folds.
%
% Output:
%   dfold:          XX
%   inds:           XX
%
% See also: sacCrossValidate, sacEvaluate, sacTrain, sacPredict

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


% TODO: does not work for leave-one-out validation (because we ensure there
% is always an example of each class in the training set)


if ~exist('folds', 'var')
    folds = 10;
end

if folds < 1
    error('Number of folds must be >= 1');
end

numClasses = numel(d.classNames);


N = size(d.instances,1);
allInstances = 1:N;
instancePool = 1:N;
inds = cell(1,folds);
trainInds = cell(1,folds);



% seed the random stream
ver = version('-release');  % get the matlab version
if str2double(ver(1:4)) >= 2009
    defaultStream = RandStream.getGlobalStream;
    savedState = defaultStream.State;
    if exist('seed', 'var')
        stream0 = RandStream('mt19937ar', 'Seed', seed);
        RandStream.setGlobalStream(stream0);
    end
else
    savedState = rand('seed');
    rand('seed', seed); %#ok<*RAND>
end



% put a minimum of 1 example of every class in each training fold
for i = 1:numClasses
    labelinds{i} = find(d.labels == i); %#ok<AGROW>
end
for k = 1:folds
    for i = 1:numClasses
        ind = randsample(labelinds{i},1);
        labelinds{i} = setdiff(labelinds{i}, ind); %#ok<AGROW>
        %instancePool = setdiff(instancePool, ind);
        trainInds{k} = [trainInds{k} ind];
    end
end
N = length(instancePool);




%% random sample to assign instances to testing
if folds > 1
    for k = 1:folds-1
        n = min( round(N/folds), length(instancePool));

        inds{k} = [inds{k} randsample( setdiff(instancePool,trainInds{k}), n)];
        instancePool = setdiff(instancePool, inds{k});

        %disp( [ num2str(length(instancePool)) '   ' num2str(n)]);

        test{k} = inds{k}; %#ok<AGROW>
        train{k} = setdiff(allInstances, inds{k}); %#ok<AGROW>
        
        % sanity check
        %[intersect(train{k}, test{k})  numel(train{k}) + numel(test{k})]
%         if sum(ismember(train{k}, trainInds{k})) == numClasses
%             disp(['fold ' num2str(k) ' training ok']);
%         end
%         if sum(ismember(test{k}, trainInds{k})) == 0
%             disp('testing ok');
%         end
    end
end
% the last iteration
k = folds;
n = min( length(instancePool), numel(setdiff(instancePool,trainInds{k})));
    
inds{k} = [inds{k} randsample(setdiff(instancePool,trainInds{k}), n)];
instancePool = setdiff(instancePool, inds{k});
%disp( [ num2str(length(instancePool)) '   ' num2str(n)]);
test{k} = inds{k}; %#ok<AGROW>
if folds > 1
    train{k} = setdiff(allInstances, inds{k}); %#ok<AGROW>
else
    train{k} = test{k}; %#ok<AGROW>
end
% sanity check
%[intersect(train{k}, test{k})  numel(train{k}) + numel(test{k})]
% if sum(ismember(train{k}, trainInds{k})) == numClasses
%     disp(['fold ' num2str(k) ' training ok']);
% end
% if sum(ismember(test{k}, trainInds{k})) == 0
%     disp('testing ok');
% end

d.trainFolds = train;
d.testFolds = test;



% restore the original randstream
if str2double(ver(1:4)) >= 2009
    defaultStream = RandStream.getGlobalStream;
    defaultStream.State = savedState;
else
    rand('seed', savedState);
end






    %[intersect(train{k}, test{k})  numel(train{k}) + numel(test{k})]
    
%     dtemp = d;
%     dtemp.fold = k;
%     dtemp.instances = dtemp.instances(inds{k},:);
%     dtemp.labels = dtemp.labels(inds{k},:);
%     dfold{k} = dtemp;
%     dfold{k}
