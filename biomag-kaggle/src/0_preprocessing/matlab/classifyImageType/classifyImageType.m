function [class,posterior]=classifyImageType(img,trainingSet,labels,varargin)
% Classify image type as one of the following:
%   - flourescent
%   - tissue
%   - brightfield
% Classification is regardless of image content, only basic intensity
% features are used (integrated-, mean-, std-, min- and max intensity).
% img:  either an image matrix or a string referring to an image name (with
%       its path included!)
% trainingSet:  feature matrix of training set
% labels:   class indexes
% classNames:   cell array of strings corresponding to class names

n=numel(unique(labels));
defCost=ones(n)-eye(n);
defPrior(1,1:n)=deal(1/n);
defClassNames=[];
allMethods={'knn','naivebayes','svm','lda','tree','ann','randomforest'};
defMethod='tree';
defClassifier=[];
defBatch=false;
defSource=pwd;

if nargin>3
    % parse additional inputs
    p=inputParser;
    
    validateFcnLabel=@(x) isnumeric(x)&min(size(squeeze(x)))==1&numel(unique(labels))>1;
    validateFcnImg=@(x) (isnumeric(x)&&min(size(squeeze(x)))>1)||(ischar(x)&&exist(x,'file'))||(isstruct(x)&&isfield(x,'name')&&numel(x)>0);
    validateFcnCellStr=@(x) iscell(x)&all(cellfun(@ischar,x));
    validateFcnMethod=@(x) ischar(x)&~isempty(find(~cellfun(@isempty,strfind(allMethods,lower(x))),1));
    addRequired(p,'img',validateFcnImg);
    addRequired(p,'trainingSet',@isnumeric);
    addRequired(p,'labels',validateFcnLabel);
    addParameter(p,'classNames',defClassNames,validateFcnCellStr);
    addParameter(p,'costMatrix',defCost,@(x) isnumeric(x)&ndims(x)==2&size(x,1)==size(x,2)&size(x,1)<=n); %#ok<ISMAT>
    addParameter(p,'prior',defPrior,@(x) isnumeric(x)&numel(x)==n&(sum(x)<=1+eps&sum(x)>=1-eps));
    addParameter(p,'method',defMethod,validateFcnMethod);
    addParameter(p,'classifier',defClassifier,@(x) ValidateFcnClassifier(x));
    addParameter(p,'batch',defBatch,@islogical);
    addParameter(p,'source',defSource,@(x) ischar(x)&exist(x,'dir'));
    
    parse(p,img,trainingSet,labels,varargin{:});
    img=p.Results.img;
    trainingSet=p.Results.trainingSet;
    labels=p.Results.labels;
    classNames=p.Results.classNames;
    costMatrix=p.Results.costMatrix;
    prior=p.Results.prior;
    method=p.Results.method;
    if ~isempty(p.Results.classifier)
        classifier=load(p.Results.classifier);
    else
        classifier=struct();
    end
    batch=p.Results.batch;
    source=p.Results.source;
end

if ischar(img)
    img=imread(img);
end

if batch
    list=img;
    for imi=1:numel(list)
        matlabVersion=regexp(version,'\(\w*\)','match');
        if strcmp(matlabVersion{1}(2:end-1),'R2017a')
            img=imread(fullfile(list(imi).folder,list(imi).name));
        else
            img=imread(fullfile(source,list(imi).name));
        end
        if size(trainingSet,2)==5
            % easy intensity features (sum, mean, std, min, max)
            img=im2double(img);
            img=img(:);

            % compute basic intensity features
            fts(imi,:)=[sum(img) mean(img) std(img) min(img) max(img)];
        else
            fts(imi,:)=extractFeaturesEasy(img);
        end
    end
else
    if size(trainingSet,2)==5
        % easy intensity features (sum, mean, std, min, max)
        img=im2double(img);
        img=img(:);

        % compute basic intensity features
        fts=[sum(img) mean(img) std(img) min(img) max(img)];
    else
        fts=extractFeaturesEasy(img);
    end
end

if ~exist('costMatrix','var') || isempty(costMatrix)
    % default costs: 0 for correct, 1 for wrong classification
    costMatrix=ones(numel(classNames))-eye(numel(classNames));
end
if ~exist('prior','var') || isempty(prior)
    % default costs: 0 for correct, 1 for wrong classification
    prior=zeros(1,numel(classNames));
    prior(:)=deal(1/numel(classNames));
end

if strcmp(method,'tree')    % default method, easy
    if isempty(fieldnames(classifier))
        % train a decision tree
        model=fitctree(trainingSet,labels,'Cost',costMatrix,'Prior',prior);
        model=prune(model);
    else
        model=classifier.model;
    end
    [classidx,posterior]=predict(model,fts);
else
    % TODO: use trainCustomClassifier()
end

if isempty(classNames)
    class=classidx;
else
    if ~batch
        class=classNames{classidx};
    else
        class=arrayfun(@(x) classNames{x},classidx,'UniformOutput',false);
    end
end

end

% alternatively, train a custom classifier
function trainCustomClassifier() %#ok<DEFNU>
    
end

% validate classifier .mat file to load
function isValid=ValidateFcnClassifier(input)
    isValid=false;
    if ~(ischar(input)&&exist(input,'file'))
        return;
    else
        tmp=matfile(input);
        fields=who(tmp);
        if any(strcmp(fields,'model'))
            isValid=true;
            return;
        end
    end
end