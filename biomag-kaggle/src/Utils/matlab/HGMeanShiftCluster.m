function [clustCent,data2cluster,cluster2dataCell] = HGMeanShiftCluster(dataPts,bandWidth,kernel,plotFlag);
%HGMEANSHIFTCLUSTER performs MeanShift Clustering of data using a chosen kernel
%
% ---INPUT---
% dataPts           - input data, (numDim x numPts)
% bandWidth         - is bandwidth parameter (scalar)
% kernel            - kernel type (flat or gaussian)
% plotFlag          - display output if 2 or 3 D    (logical)
% ---OUTPUT---
% clustCent         - is locations of cluster centers (numDim x numClust)
% data2cluster      - for every data point which cluster it belongs to (numPts)
% cluster2dataCell  - for every cluster which points are in it (numClust)
% 
% Copyright 2015 Han Gong, University of East Anglia
% Copyright 2006 Bart Finkston
%
% MeanShift first appears in
% K. Funkunaga and L.D. Hosteler, "The Estimation of the Gradient of a
% Density Function, with Applications in Pattern Recognition"


%*** Check input ****
if nargin < 2
    error('no bandwidth specified')
end

if nargin < 4
    plotFlag = false;
end

%**** Initialize stuff ***
[numDim,numPts] = size(dataPts);
numClust = 0;
bandSq = bandWidth^2;
initPtInds = 1:numPts;
maxPos = max(dataPts,[],2); % biggest size in each dimension
minPos = min(dataPts,[],2); % smallest size in each dimension
boundBox = maxPos-minPos; % bounding box size
sizeSpace = norm(boundBox); % indicator of size of data space
stopThresh = 1e-3*bandWidth; % when mean has converged
clustCent = []; % center of clust
beenVisited= false(1,numPts); % track if a points been seen already
numInitPts = numPts; % number of points to posibaly use as initilization points
clusterVotes = zeros(1,numPts,'uint16'); % used to resolve conflicts on cluster membership
clustMembsCell = [];

%*** mean function with the chosen kernel ****
switch kernel
case 'flat' % flat kernel
    kmean = @(x,dis) mean(x,2);
case 'gaussian' % approximated gaussian kernel
    kmean = @(x,d) gaussfun(x,d,bandWidth);
otherwise
    error('unknown kernel type');
end

while numInitPts
    tempInd = ceil( (numInitPts-1e-6)*rand); % pick a random seed point
    stInd = initPtInds(tempInd); % use this point as start of mean
    myMean = dataPts(:,stInd);  % intilize mean to this points location
    myMembers = []; % points that will get added to this cluster                          
    thisClusterVotes = zeros(1,numPts,'uint16'); % used to resolve conflicts on cluster membership

    while true %loop untill convergence
        sqDistToAll = sum(bsxfun(@minus,myMean,dataPts).^2); % dist squared from mean to all points still active
        
        inInds = find(sqDistToAll < bandSq); % points within bandWidth
        thisClusterVotes(inInds) = thisClusterVotes(inInds)+1; % add a vote for all the in points belonging to this cluster
        
        myOldMean = myMean; % save the old mean
        myMean = kmean(dataPts(:,inInds),sqrt(sqDistToAll(inInds))); % compute the new mean
        myMembers = [myMembers inInds]; % add any point within bandWidth to the cluster
        beenVisited(myMembers) = true; % mark that these points have been visited
        
        %*** plot stuff ****
        if plotFlag
            figure(12345),clf,hold on
            if numDim == 2
                plot(dataPts(1,:),dataPts(2,:),'.')
                plot(dataPts(1,myMembers),dataPts(2,myMembers),'ys')
                plot(myMean(1),myMean(2),'go')
                plot(myOldMean(1),myOldMean(2),'rd')
                pause(0.1);
            end
        end

        %**** if mean doesn't move much stop this cluster ***
        if norm(myMean-myOldMean) < stopThresh
            %check for merge posibilities
            mergeWith = 0;
            for cN = 1:numClust
                distToOther = norm(myMean-clustCent(:,cN)); % distance to old clust max
                if distToOther < bandWidth/2 % if its within bandwidth/2 merge new and old
                    mergeWith = cN;
                    break;
                end
            end
            
            if mergeWith > 0 % something to merge
                nc = numel(myMembers); % current cluster's member number
                no = numel(clustMembsCell{mergeWith}); % old cluster's member number
                nw = [nc;no]/(nc+no); % weights for merging mean
                clustMembsCell{mergeWith} = unique([clustMembsCell{mergeWith},myMembers]);   %record which points inside 
                clustCent(:,mergeWith) = myMean*nw(1) + myOldMean*nw(2);
                clusterVotes(mergeWith,:) = clusterVotes(mergeWith,:) + thisClusterVotes;    %add these votes to the merged cluster
            else % it's a new cluster
                numClust = numClust+1; %increment clusters
                clustCent(:,numClust) = myMean; %record the mean  
                clustMembsCell{numClust} = myMembers; %store my members
                clusterVotes(numClust,:) = thisClusterVotes; % creates a new vote
            end

            break;
        end

    end
    
    initPtInds = find(~beenVisited); % we can initialize with any of the points not yet visited
    numInitPts = length(initPtInds); %number of active points in set
end

[~,data2cluster] = max(clusterVotes,[],1); % a point belongs to the cluster with the most votes

%*** If they want the cluster2data cell find it for them
if nargout > 2
    cluster2dataCell = cell(numClust,1);
    for cN = 1:numClust
        myMembers = find(data2cluster == cN);
        cluster2dataCell{cN} = myMembers;
    end
end

end
