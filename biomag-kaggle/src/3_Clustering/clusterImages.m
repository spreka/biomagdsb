function targetFile = clusterImages( DD, imageName, targetDir,type)
%clusterImages
%   Clusters the images based on the Distance matrix stored in DD (the rows
%   corresponds to imageNames)
%   targetDir a csv filepath that will be created for the
%   clusters. Name is generated automatically
%   type selection

fileName = ['predictedStyles_' type '.csv'];
if ~exist(targetDir,'dir'), mkdir(targetDir); end
targetFile = fullfile(targetDir,fileName);
f = fopen(targetFile,'w');
fprintf(f,'Name,Style\n');

nofClusters = 70;
fprintf('Clustering images...\n');
DD = squareform(DD);

switch type
    case 'Kmeans-Basic'        
        [idx] = kmeans(DD,nofClusters);
    case 'Kmeans-adaptive'
        fprintf('Adaptive Kmeans clustering\n');
        nofImgsPerCluster = 10:2:20;
        N = size(DD,1);
        initNumber = round(N./nofImgsPerCluster);
        nofTrials = length(initNumber);
        sums = zeros(1,nofTrials);
        clusterCandidates = cell(1,nofTrials);
        for i = 1:nofTrials
            fprintf('Iteration %d/%d\n',i,nofTrials);
            [clusterCandidates{i},~,clusterD] = kmeans(DD,initNumber(i),'Replicates',30,'MaxIter',10000,'Distance','correlation');
            sums(i) = sum(clusterD);
        end
        [~,mini] = min(sums);
        idx = clusterCandidates{mini};
    case 'Hierarchical-single'        
        Z = linkage(DD,'single','correlation');
        idx = cluster(Z,'maxclust',nofClusters);
    case 'Kmeans-cosine'
        [idx] = kmeans(DD,nofClusters,'Replicates',20,'Distance','cosine');
    case 'Kmeans-correlation-Best3Cluster'        
        fprintf('Adaptive Kmeans clustering\n');
        nofImgsPerCluster = 5:1:10;
        N = size(DD,1);
        initNumber = round(N./nofImgsPerCluster);
	initNumber(initNumber==0)=1;
	nofTrials = length(initNumber); %<-- fails on small number of input images
        %nofTrials = length(find(initNumber>0));
        sums = zeros(1,nofTrials);
        clusterCandidates = cell(1,nofTrials);
        D = cell(1,nofTrials);
        for i = 1:nofTrials
            fprintf('Iteration %d/%d\n',i,nofTrials);
            [clusterCandidates{i},~,clusterD,D{i}] = kmeans(DD,initNumber(i),'Replicates',30,'MaxIter',10000,'Distance','correlation');
            sums(i) = sum(clusterD);
        end
        [~,mini] = min(sums);        
        idx = clusterCandidates{mini};
        idx = filterBest(D{mini},idx,initNumber(mini),3);
     case 'Kmeans-cosine-Best5Cluster'        
         fprintf('Adaptive Kmeans clustering\n');
         nofImgsPerCluster = 10:2:20;
         N = size(DD,1);
         initNumber = round(N./nofImgsPerCluster);
         nofTrials = length(initNumber);
         sums = zeros(1,nofTrials);
         clusterCandidates = cell(1,nofTrials);
         D = cell(1,nofTrials);
         for i = 1:nofTrials
             fprintf('Iteration %d/%d\n',i,nofTrials);
             [clusterCandidates{i},~,clusterD,D{i}] = kmeans(DD,initNumber(i),'Replicates',30,'MaxIter',10000,'Distance','cosine');
             sums(i) = sum(clusterD);
         end
         [~,mini] = min(sums);
         idx = clusterCandidates{mini};
         idx = filterBest(D{mini},idx,initNumber(mini),5);
    case 'Kmeans-correlation-Best5Cluster'
        fprintf('Adaptive Kmeans clustering best 5\n');
        nofImgsPerCluster = 20:2:30;
        N = size(DD,1);
        initNumber = round(N./nofImgsPerCluster);
        nofTrials = length(initNumber);
        sums = zeros(1,nofTrials);
        clusterCandidates = cell(1,nofTrials);
        D = cell(1,nofTrials);
        for i = 1:nofTrials
            fprintf('Iteration %d/%d\n',i,nofTrials);
            [clusterCandidates{i},~,clusterD,D{i}] = kmeans(DD,initNumber(i),'Replicates',50,'MaxIter',10000,'Distance','correlation');
            sums(i) = sum(clusterD);
        end
        [~,mini] = min(sums);
        idx = clusterCandidates{mini};
        idx = filterBest(D{mini},idx,initNumber(mini),5);
    case 'Kmeans-correlation-Best3Cluster-LESS-ALL' % previously named only with LESS
        fprintf('Adaptive Kmeans clustering\n');
        nofImgsPerCluster = 10:2:20;
        N = size(DD,1);
        initNumber = round(N./nofImgsPerCluster);
        nofTrials = length(initNumber);
        sums = zeros(1,nofTrials);
        clusterCandidates = cell(1,nofTrials);
        D = cell(1,nofTrials);
        for i = 1:nofTrials
            fprintf('Iteration %d/%d\n',i,nofTrials);
            [clusterCandidates{i},~,clusterD,D{i}] = kmeans(DD,initNumber(i),'Replicates',30,'MaxIter',10000,'Distance','correlation');
            sums(i) = sum(clusterD);
        end
        [~,mini] = min(sums);
        idx = clusterCandidates{mini};
        idx = filterBest(D{mini},idx,initNumber(mini),3);
end

for i=1:length(imageName)
    fprintf(f,'%s,%d\n',imageName{i},idx(i));
end


fclose(f);

end

function updatedIdx = filterBest(distM,idx,k,nofBest)
    updatedIdx = idx;
    for i=1:k
        thisClusterID = find(idx == i);
        distance = distM(thisClusterID,i);
        [~,sortedIdx] = sort(distance);
        updatedIdx(thisClusterID(sortedIdx(nofBest+1:end))) = -1;
    end
end

