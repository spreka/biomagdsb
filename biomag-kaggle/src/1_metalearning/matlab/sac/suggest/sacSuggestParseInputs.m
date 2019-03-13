function [timeLimit,searchMethod,pll,classifiers,M,filenm] = sacSuggestParseInputs(v)
% parse the inputs find: timeLimit    - search time for a given classifier
%                        searchMethod - the type of search to use
%                        pll          - 1=distributed mode, 0=serial mode
%                        classifiers  - the list of classifers to search
%                        measures
%                        weights
%                        criteria

global logFID
global displayMessagePriority
global logMessagePriority


% default sacSuggest options
timeLimit = 60;
searchMethod = 'random';
pll = sacCheckDistributedToolbox();

% the default classifiers
defaultClassifiers = {'BayesNet_Weka', 'BoostedTree_Weka', 'DecisionTree_C45_Weka', 'KNN_Weka', 'LogisticRegression_Weka', 'LogitBoost_Weka', 'MLP_Weka', 'NaiveBayes_Weka', 'NearestNeighbor_Weka', 'RandomForest_Weka', 'SVM_Libsvm', 'SVM_Weka'};
%classifiers = {'KNN_Weka', 'RandomForest_Weka', 'SVM_Weka', 'SVM_Libsvm', 'MLP_Weka', 'LogitBoost_Weka', 'Logistic_Weka'};

displayMessagePriority = 3;
logMessagePriority = 5;
classifiers = {};
measures = {};
criteria  ={};
weights = {};
filenm = 'SACsearch.mat';

% get a list of supported classifiers
% TODO: use a check on the system to find valid classifiers
classifierList = lower(sacGetSupportedList('classifiers'));
metricList = upper(sacGetSupportedList('metrics'));
criteriaList = upper(sacGetSupportedList('criteria'));

for i = 1:length(v)
    % if the input argument is a string
    if ischar(v{i})
        switch lower(v{i})
            case 'timelimit'
                timeLimit = v{i+1};
            case {'random', 'bruteforce', 'default', 'sa', 'simulatedannealing'}; %TODO: update when more are avail.
                searchMethod = v{i};
            case 'displaymessagepriority'
                displayMessagePriority = v{i+1};
            case 'logmessagepriority'  
                logMessagePriority = v{i+1};
            case {'parallel', 'distributed'}
                pll = 1;
            case {'nonparallel', 'serial'}
                pll = 0;
            case classifierList  
                classifiers = v(i); % even a single classifier should be put into a cell
            case 'criteria'
                if ischar(v{i+1})
                    criteria = v(i+1);
                else
                    criteria = v{i+1};
                end
            case 'measures'
                if ischar(v{i+1})
                    measures = v(i+1);
                else
                    measures = v{i+1};
                end
            case 'weights'
                if isnumeric(v{i+1})
                    weights = v(i+1);
                else
                    weights = v{i+1};
                end
            case 'filename'
                filenm = v{i+1};
            otherwise
                if ~ismember(upper(v{i}), metricList)
                    if numel(v) > 1
                        if ~strcmpi(v{i-1}, 'filename')
                            error(['Unknown input ' v{i} '.']);
                        end
                    end
                end
        end
    end
    
    % if the input argument is a cell
    if iscell(v{i})
        % check to see if v{i} is a list of metrics or classifiers
        if ischar(v{i}{1})
            if ismember(lower(v{i}{1}), classifierList)
                classIn = v{i};
                for j = 1:length(classIn)
                    if ismember(lower(classIn{j}), classifierList)
                        classifiers{end+1} = classIn{j}; %#ok<AGROW>
                    end
                end
                
            end
        end
    end
end


% handle case when measures, weights, or criteria are not specified
if isempty(measures) && isempty(criteria)
    measures = {'ACC','AUC','BEP','FSC','LFT','RMS'};
    criteria = measures;
elseif isempty(measures) && ~isempty(criteria)
    measures = criteria;
elseif ~isempty(measures) && isempty(criteria)
    criteria = measures;
end

measures = upper(measures);
criteria = upper(criteria);


% 1. if weights are given, decide if they correspond to measures or criteria
if ~isempty(weights)
    if numel(weights) == numel(measures)
        measureWeights = 1;
    elseif numel(weights) == numel(criteria)
        measureWeights = 0;
    else
        error('Number of weights does not match number of measures.');
    end
end


% 2. make sure measures and criteria are valid
for m = 1:length(measures)
    if ~ismember(measures{m}, metricList)
        error(['Unknown metric ' measures{m} ' specified.']);
    end
end
for m = 1:length(criteria)
    if ~ismember(criteria{m}, criteriaList)
        criteria(m) = [];
        if ~isempty(weights) && ~measureWeights
            weights(m) = []; %#ok<AGROW>
        end
    end
end


% 3. assign weights to each measure. the input weights may be associated 
% with the criteria or the measures
if ~isempty(weights)
    if measureWeights
        % if the weights provided correspond to the measures
        numProvidedMeasures = numel(measures);
        missingMeasures = setdiff(criteria, measures);
        measures = [measures, missingMeasures];
        for m = 1:numProvidedMeasures
            if isempty(weights{m})
                weights{m} = 1; %#ok<AGROW>
            end
        end
        if ~isempty(missingMeasures)
            for m = numel(measures):-1:(numel(measures)-numel(missingMeasures)+1)
                weights{m} = 1; 
            end
        end
    else    
        % if the weights provided correspond to the criteria
        weightsIn = weights;
        missingMeasures = setdiff(criteria, measures);
        measures = [measures, missingMeasures];
        for m = 1:numel(measures)
            [tf ind] = ismember(measures{m}, criteria);
            if tf
                if isempty(weightsIn{ind})
                    weights{m} = 1; %#ok<AGROW>
                else
                    weights{m} = weightsIn{ind}; %#ok<AGROW>
                end
            else
                weights{m} = 1; %#ok<AGROW>
            end
        end
    end
    % sort
    [measures,inds] = sort(measures);
    weights = weights(inds);
    M.weights = weights;
end

M.measures = measures;
M.criteria = criteria;


% 4. ensure that we have selected some valid classifiers
while isempty(classifiers)
    classIn = input(' No valid classifiers were specified, please enter a list (cell) of classifier strings\n or press ENTER to use the default list of classifiers:\n');
    if ~iscell(classIn)
        classifiers = defaultClassifiers;
    else
        for i = 1:length(classIn)
            if ismember(lower(classIn{i}), classifierList)
                classifiers{end+1} = classIn{i}; %#ok<AGROW>
            end
        end
    end
end






% Output to the log/command our search parameters!

% check for distributed mode
if pll == 1
    pllstr = 'Distributed';
    sched = findResource('scheduler', 'type', 'local');       % job manager
    clusterSize = sched.ClusterSize;    % number of available workers  % TODO: allow non-local job managers
else
    pllstr = 'Serial';
    clusterSize = 1;
end


% other various options
sacLog(3,logFID,' - - - - - - - - - - - - - - - - - - - - - - - - - -\n');
sacLog(3,logFID,' Methods to search:\n');
for a = 1:length(classifiers)
    sacLog(3,logFID, '   %s\n', classifiers{a}); %#ok<AGROW>
end
sacLog(3,logFID,' Using %s search method.\n', upper(searchMethod));
sacLog(3,logFID,' %s processing mode (%d CPUs).\n', pllstr, clusterSize);
sacLog(3,logFID,' %d second time limit per method.\n', timeLimit);
sacLog(3,logFID,' Performance-Index metrics:');
for a = 1:length(M.criteria)
    sacLog(3,logFID, ' %s', M.criteria{a}); %#ok<AGROW>
end
sacLog(3,logFID,'.\n - - - - - - - - - - - - - - - - - - - - - - - - - -\n');
    

