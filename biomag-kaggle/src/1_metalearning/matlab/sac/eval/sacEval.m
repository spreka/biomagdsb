function [E s] = sacEval(Y, p, varargin)
% Y = ground truth labels
% p = numInstances x numClasses matrix containing class predictions
%
%
% if the dataset is given as one of the inputs, we can extract the
% classnames and the dataset name
%
% weights for each of the metrics can also be provided as a cell, it must
% appear in the same order as the metrics. ROC and PR do not need
% associated weights, but the weight cell must contain entries for them to
% keep the proper ordering
%
% if 'criteria' is included, will compute the performance criteria
%
% the size of the weights must match the size of metrics, not criteria
% [E s] = sacEval(d.labels, p, d, {'AUC','ACC','BEP','CONFUSION','FPR','RMS','PR','TPR'},'criteria', {'AUC','ACC','BEP','CONFUSION','FPR','RMS','PR'},'weights', {1,1,2,1,1,-1,[],1});
% [E s] = sacEval(d.labels, p, d, {'AUC','ACC','BEP','CONFUSION','FPR','RMS','PR','TPR'},'criteria', {'AUC','ACC','BEP'},'weights', {1,1,2,1,1,-1,[],1});
% [E s] = sacEval(d.labels,p,d,M);  % M is a metric structure


% determine the number of classes
numClasses = size(p,2);

% initialize the data structure
E = initE(); E.numInstances = numel(Y);

% parse the inputs to get the data set name, the class names, and the set
% of metrics to be evaluated, the metric weights, and the performance 
% criteria (if specified)
[E metrics quietMode emptyMode] = parseInput(varargin, E, numClasses, Y);

% if the user requested an "empty" data structure, return
if emptyMode; s = []; return; end

% TODO: check for unpredicted entries in 'p'
p = p ./ repmat(sum(p,2), 1, size(p,2));  % normalize the probabilities
psum = sum(p,2);
%unpredicted = find(psum ~= 1);
% fill in any unpredicted entries in p with uniform probabilities
unpredicted = find(isnan(psum));
for i = 1:length(unpredicted)
    p(unpredicted(i),:) = ones(1, size(p,2)) ./ size(p,2);
end


% ----------------- COMPUTE THE SPECIFIED METRICS --------------------


% compute GLA, GLE, CORRECT, INCORRECT
if ~isempty(intersect(metrics, {'ACC','ERR','CORRECT','INCORRECT'}))
    y = scores2Labels(p);
    E.CORRECT   = sum((y == Y));
    E.INCORRECT = sum((y ~= Y));
    E.ACC = E.CORRECT / E.numInstances;
    E.ERR = 1 - E.ACC;
end

% compute PRE, REC, FSC,FPR,TPR,CONFUSION. Also computed this if ROC is 
% requested, we also compute these so we can see actual operating points
if ~isempty(intersect(metrics, {'PRE','REC','FSC','FPR','CONFUSION','ROC','TPR'}))
    [E.PRE E.REC E.TPR E.FPR E.FSC E.CONFUSION] = computeBasicMeasures(numClasses, Y, p);
end

% compute RMS
if ~isempty(intersect(metrics, {'RMS'}))
    E.RMS = computeRMS(Y,p); 
end

% compute MXE
if ~isempty(intersect(metrics, {'MXE'}))
    E.MXE = computeMXE(Y,p);
end

% compute LFT
if ~isempty(intersect(metrics, {'LFT'}))
    E.LFT = computeLift(Y,p);
end

% compute PRE, REC, FSC, 
if ~isempty(intersect(metrics, {'ROC','AUC','PR','BEP'}))
    [E.ROC E.AUC E.PR E.BEP] = sacRankMetrics(Y, p);
end

% compute the performance criteria, if necessary
if ~isempty(E.performanceCriteria)
    E.performanceValues = computePerformanceValues(E);
end

% If requested, construct the output string and display it 
if ~quietMode
    s = sacEval2String(E, metrics);
    disp(s);
else
    s =[];
end


%keyboard;










% ------------------------ SUPPORTING FUNCTIONS -----------------------


function y = scores2Labels(p)
% Converts a class-wise matrix of size NxM (N instances, M classes) into an Nx1
% vector containing the predicted class indexes.

[dummy, y] = max(p, [] ,2); %#ok<ASGLU>
clear dummy;



function E = initE()

% set the fields of the evaluation structure E to be empty
E.classNames            = [];
E.dataset               = [];
E.numClasses            = [];
E.numInstances          = [];
E.performanceCriteria   = [];
E.performanceCoeffs     = [];
E.performanceValues     = [];
E.performanceCalibrated = [];
E.predictionTime        = [];
E.trainingTime          = [];
E.ACC                   = [];
E.AUC                   = [];
E.BEP                   = [];
E.CONFUSION             = [];
E.CORRECT               = [];
E.ERR                   = [];
E.FSC                   = [];
E.FPR                   = [];
E.INCORRECT             = [];
E.LFT                   = [];
E.MXE                   = [];
E.PR                    = [];
E.PRE                   = [];
E.REC                   = [];
E.RMS                   = [];
E.ROC                   = [];
E.TPR                   = [];
E.Wacc                  = [];
E.Wauc                  = [];
E.Wbep                  = [];
E.Wconfusion            = [];
E.Wcorrect              = [];
E.Werr                  = [];
E.Wfsc                  = [];
E.Wfpr                  = [];
E.Wincorrect            = [];
E.Wlft                  = [];
E.Wmxe                  = [];
E.Wpre                  = [];
E.Wrec                  = [];
E.Wrms                  = [];
E.Wtpr                  = [];


%function [dName classNames metrics quietMode emptyMode classWeights] = parseInput(v, numClasses, Y)
function [E metrics quietMode emptyMode] = parseInput(v, E, numClasses, Y)

metrics = {};
quietMode = 0;
emptyMode = 0;
weights = [];
criteria = {};

% set some flags to be used later
datasetGiven = 0;
weightsGiven = 0;
criteriaGiven = 0;

metricList = upper(sacGetSupportedList('metrics'));

for i=1:length(v)
    if ischar(v{i})
        switch upper(v{i})
            case metricList
                metrics = v(i);
            case 'WEIGHTS'
                if iscell(v{i+1}) || isnumeric(v{i+1})
                    weightsGiven = 1;
                    weights = v{i+1};
                end
            case 'CRITERIA'
                if iscell(v{i+1}) || ischar(v{i+1})
                    criteriaGiven = 1;
                    criteria = v{i+1};
                end
            case 'EMPTY'
                emptyMode = 1;
                % do nothing
            case 'QUIET'
                quietMode = 1;
            otherwise
                error(['Unknown input ' v{i} '.']);
        end               
    elseif iscell(v{i})
        if i > 1
            if ischar(v{i-1})
                if ismember(v{i-1}, {'WEIGHTS', 'CRITERIA'})
                    % do nothing
                end
            elseif ismember(v{i}{1}, metricList)
                metrics = v{i};
            end
        else
            % the cell contains the metrics
            metrics = v{i};
        end
    elseif isstruct(v{i})
        if isfield(v{i}, 'instances')
            % the structure dataset structure was given
            d = v{i};
            classNames = d.classNames;
            datasetGiven = 1;    
        elseif isfield(v{i}, 'measures')
            % a metrics structure was given
            metrics = v{i}.measures;
            if isfield(v{i}, 'criteria')
                if ~isempty(v{i}.criteria)
                    criteriaGiven = 1;
                    criteria = v{i}.criteria;
                end
            end
            if isfield(v{i}, 'weights')
             	weightsGiven = 1;
                weights = v{i}.weights;
            end
        end
    end
end

if isempty(metrics) 
    if criteriaGiven
        metrics = criteria;
    elseif emptyMode
        %return;
    else
        error('Metrics / criteria not specified');
    end
end
    

metrics = sort(metrics);
for m = 1:numel(metrics)
    if ~ismember(metrics{m},metricList)
        error(['Unknown metric ' metrics{m} ' specified.']);
    end
end



% assign a name to the dataset
if datasetGiven
    dName = d.name;
else
    dName = 'Unspecified Dataset';
end

% assign class names
if datasetGiven
    classNames = d.classNames;
else
    if numClasses > 0
        for n = 1:numClasses
            classNames{n} = ['Class ' num2str(n)];
        end
    else
        classNames = [];
    end
end

if isempty(Y)
    if datasetGiven
        E.numInstances = numel(d.labels);
    end
else
    E.numInstances = numel(Y);
end

% determine default class weights (how often each class appears in the data)
if numClasses > 0
    classWeights = zeros(1,numClasses);
    if datasetGiven
        % by default, classes are weighted according to the frequency they 
        % appear in the dataset, if the dataset is given
        %numInstances = size(Y,1);
        for i = 1:numClasses
            classWeights(i) = sum(Y == i) / E.numInstances;
        end
        classWeights = classWeights / sum(classWeights);
    else
        % otherwise, we make each class of equal weight
        classWeights = ones(1,numClasses)/numClasses;
    end
else
    classWeights = [];
end


% initialize the data structure
E.classNames = classNames; E.dataset = dName; %E.classWeights = classWeights;
E.numClasses = numClasses; 


if (E.numClasses == 0) && datasetGiven
    E.numClasses = numel(d.classNames);
end

if emptyMode
    return;
end

E = setMetricWeights(E, metrics, classWeights, weightsGiven, weights);

criteriaList = sacGetSupportedList('criteria');
if criteriaGiven
    E.performanceCriteria = intersect(criteria, criteriaList);
else
    E.performanceCriteria = intersect(metrics, criteriaList);
end



% Determine the performance coefficients (how relatively important each of
% the metrics is in the PerformanceIndex).
%
E.performanceCoeffs = zeros(1,length(E.performanceCriteria));
for m = 1:length(E.performanceCriteria)
    criteriaField = E.performanceCriteria{m};
    weightField = ['W' lower(criteriaField)];
    
    if ~isequal(criteriaField, 'CONFUSION')
        n = sum(E.(weightField));
        E.(weightField) = E.(weightField)/n;
        E.performanceCoeffs(m) = n;
    else
        % confusion matrix is handled differently     
        E.performanceCoeffs(m) = max(max(E.(weightField)));
    end
    
end

% rescale the performance coefficients so that the abs values sum to 1
E.performanceCoeffs = E.performanceCoeffs ./ sum(abs(E.performanceCoeffs));



function E = setMetricWeights(E, metrics, classWeights, weightFlag, weights)

if weightFlag
    if numel(metrics) ~= numel(weights)
        error('Size of weight vector does not match size of metrics');
    end
end


% if weights were specified, interpret them and assign them to fields in E
if weightFlag
    for m = 1:numel(metrics)
        switch metrics{m}
            case {'AUC','BEP','FSC','LFT','PRE','REC','TPR'}
                mName = ['W' lower(metrics{m})];
                w = weights{m};
                if numel(w) == 1
                    % we have been given a scalar to factor the
                    % classWeights by
                    E.(mName) = w * classWeights;
              	elseif numel(w) == 0
                    E.(mName) = classWeights;
                else
                    % we have been given the weight directly
                    if numel(w) == numel(classWeights)  %isequal(size(w), size(classWeights))
                        E.(mName) = w;
                    else
                        error(['Incorrect dimensionality for metric ' metrics{m} ' weight, expected ' num2str(size(classWeights,1)) ' x ' num2str(size(classWeights,2))]);
                    end
                end
            case {'FPR'}
                % lower values are better for these metrics, so multiply -1
                mName = ['W' lower(metrics{m})];
                w = weights{m};
                if numel(w) == 1
                    % we have been given a scalar to factor the
                    % classWeights by
                    if w > 0
                        w = -w;
                    end
                    E.(mName) = w * classWeights;
                elseif numel(w) == 0
                    E.(mName) = -1*classWeights;
                else
                    % we have been given the weight directly
                    if numel(w) == numel(classWeights)  %isequal(size(w), size(classWeights))
                        E.(mName) = w;
                    else
                        error(['Incorrect dimensionality for metric ' metrics{m} ' weight, expected ' num2str(size(classWeights,1)) ' x ' num2str(size(classWeights,2))]);
                    end
                end
            case 'CONFUSION'
                % if not otherwise specified, the confusion matrix should
                % have positive values on the diagonal, and negative values
                % in the upper and lower triangles
                mName = ['W' lower(metrics{m})];
                w = weights{m};
                if numel(w) == 1
                    %a = -1*ones(E.numClasses) + 3*eye(E.numClasses);
                    a = eye(E.numClasses);
                    %w = w * (a./E.numInstances);
                    E.(mName) = a;
                elseif numel(w) == 0
                    %a = -1*ones(E.numClasses) + 3*eye(E.numClasses);
                    a = eye(E.numClasses);
                    %w = 1 * (a./E.numInstances);
                    E.(mName) = a;
                else
                    if isequal(size(w), [E.numClasses,E.numClasses])
                        E.(mName) = w;
                    else
                        error(['Incorrect dimensionality for metric ' metrics{m} ' weight, expected ' num2str(numClasses) ' x ' num2str(numClasses)]);
                    end
                end
            case {'ACC', 'CORRECT'}
                mName = ['W' lower(metrics{m})];
                w = weights{m};
                if numel(w) == 1
                    if w > 0 
                        E.(mName) = w;
                    else
                        E.(mName) = -w;
                    end
             	elseif numel(w) == 0
                    E.(mName) = 1;
                else
                    error(['Incorrect dimensionality for metric ' metrics{m} ' weight, expected a scalar value']);
                end
            case {'INCORRECT','ERR','MXE','RMS'}
                mName = ['W' lower(metrics{m})];
                w = weights{m};
                if numel(w) == 1
                    if w < 0 
                        E.(mName) = w;
                    else
                        E.(mName) = -w;
                    end
                elseif numel(w) == 0
                    E.(mName) = -1;
                else
                    error(['Incorrect dimensionality for metric ' metrics{m} ' weight, expected a scalar value']);
                end
            case {'PR','ROC'}
                % there are no weights associated with PR and ROC curves
        end
    end
    
% otherwise, put in default weights (use classWeights in most cases)
else
    for m = 1:numel(metrics)
        switch metrics{m}
            case {'AUC','BEP','FSC','LFT','PRE','REC','TPR'}
                % if no weight has been specified, assign classweights to
                % these metric weight vectors (each should sum to 1)
                mName = ['W' lower(metrics{m})];
                E.(mName) = classWeights;
            case {'FPR'}
                % lower values are better for these metrics, so multiply -1
                mName = ['W' lower(metrics{m})];
                E.(mName) = -1*classWeights;
            case 'CONFUSION'
                % if not otherwise specified, the confusion matrix should
                % have positive values on the diagonal, and negative values
                % in the upper and lower triangles
                mName = ['W' lower(metrics{m})];
                %w = -1*ones(E.numClasses) + 3*eye(E.numClasses);
                w = eye(E.numClasses);
                %w = w./E.numInstances;
                E.(mName) = w;
            case {'ACC', 'CORRECT'}
                mName = ['W' lower(metrics{m})];
                E.(mName) = 1;
            case {'INCORRECT','ERR','MXE','RMS'}
                mName = ['W' lower(metrics{m})];
                E.(mName) = -1;
            case {'PR','ROC'}
                % there are no weights associated with PR and ROC curves
        end
    end
end

%keyboard;


function RMS = computeRMS(Y,p)
% worst-case RMS is 1 (all classes predicted wrongly with 100% confidence)
% best-case RMS is 0 (all classes predicted correctly with 100% confidence)



numInstances = numel(Y); %size(p,1);
numClasses = size(p,2);

Y = sacLabels2All(Y);

err = zeros(numInstances,1);
for i = 1:numInstances
    err(i) = sqrt(  sum( (Y(i,:)- p(i,:)).^2 ) /numClasses );
end

RMS = (1/numInstances) * sum(err);




function LIFT = computeLift(Y,p)
%
%

percentPopulation = .25;

numClasses = size(p,2);
numInstances = size(p,1);
Y = sacLabels2All(Y);

LIFT = zeros(1,numClasses);
for m = 1:numClasses
    clear pdescend ldescend
    
    % sort the probabilities and associated labels (descending values)
    [pdescend, inds] = sort(p(:,m), 'descend'); %#ok<ASGLU>
    ydescend = Y(inds,m);
    
    lastInd = round(percentPopulation * numInstances);
    
    %pPop = pdescend(1:lastInd);
    yPop = ydescend(1:lastInd);
    
    LIFT(m) =  (sum(yPop) / numel(yPop)) / percentPopulation;
    
end




function MXE = computeMXE(Y,p)
% worst-case MXE is INF (classes predicted wrongly with 100% confidence)
% best-case MXE is 0 (all classes predicted with 100% confidence)

% TODO: should be adjust if the ground truth is uncertain (probability is
%       not just 0 or 1)


maxEntropy = log(.001);

ent = zeros(size(Y));
for i = 1:numel(Y)
    ent(i) = (1 * log(p(i,Y(i))));    
end

infInds = isinf(ent);
if ~isempty(infInds)
    ent(infInds) = maxEntropy;
    disp('Warning: maximum entropy was used!');
end
MXE = -1/numel(Y) * sum(ent);




function [PRE REC TPR FPR FSC CONFUSION] = computeBasicMeasures(numClasses, Y, p)

PRE = zeros(1,numClasses);
REC = PRE;
FSC = PRE;
TP = PRE; FP = PRE; TN = PRE; FN = PRE; TPR = PRE; FPR = PRE;
CONFUSION = zeros(numClasses,numClasses);

P = sacAll2Labels(p);

numInstances = numel(Y);

POS = zeros(1,numClasses); NEG = POS;
for m = 1:numClasses
    POS(m) = sum(Y == m);
    NEG(m) = numInstances - POS(m);
end

for i = 1:numInstances
    otherClasses = 1:numClasses;
    
    if Y(i) == P(i)
        otherClasses(Y(i)) = [];
        TP(Y(i)) = TP(Y(i)) + 1;
        TN(otherClasses) = TN(otherClasses) + 1;
    else
        FP(P(i)) = FP(P(i)) + 1;
        FN(Y(i)) = FN(Y(i)) + 1;
        otherClasses([Y(i) P(i)]) = [];
        TN(otherClasses) = TN(otherClasses) + 1;
    end
        
    CONFUSION(Y(i),P(i)) = CONFUSION(Y(i),P(i)) + 1;
end

for m = 1:numClasses
    PRE(m) = TP(m) / (TP(m) + FP(m));
    REC(m) = TP(m) / POS(m);
    TPR(m) = TP(m) / POS(m);
    FPR(m) = FP(m) / NEG(m);
    
    if isnan(PRE(m))
        PRE(m) = 0;
    end
    
    FSC(m) = 2 * ( (1/PRE(m) ) + (1/REC(m)))^(-1);
end


function M = computePerformanceValues(E)

M = zeros(1,numel(E.performanceCriteria));
for m = 1:numel(E.performanceCriteria)
    criteriaField = E.performanceCriteria{m};
    weightField = ['W' lower(criteriaField)];
    
    if ~isequal(criteriaField, 'CONFUSION')
        M(m) = E.(weightField) * E.(criteriaField)';
    else
        % compute the confusion slightly differently
        M(m) = sum(sum(E.(weightField) .* E.(criteriaField))) / (E.numInstances);
    end
    
end











% % assign weights to each class
% if ~exist('classWeights', 'var')
%     if numClasses > 0 
%         % if classWeights are not given, classes are weighted according to
%         % frequency they appear in the dataset
%         numInstances = size(Y,1);
%         for i = 1:numClasses
%             classWeights(i) = sum(Y == i) / numInstances;
%         end
%         classWeights = classWeights / sum(classWeights);
%     else
%         classWeights = [];
%     end
% end
% 
% % TODO: add an option to make each class to be equally important
% % if isempty(E.classWeights)
% %     for i = 1:numClasses
% %         E.classWeights(i) = numInstances / sum(Y == i);
% %     end
% %     E.classWeights = E.classWeights / sum(E.classWeights);
% % end




