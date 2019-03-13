function prepareClustering4singleClusterFcn(mergedImagesDir,initialSegmentation,...
    clusterDir,styleTransTrainDir,splitOptionsFile,masks2generate,objType)
% prepare clustering and mask generationó for single-cluster images (one 
% experiment only)

if nargin==0
    mergedImagesDir='/new_hdd/spreka/cellbody/correctInput/images/';
    initialSegmentation='/new_hdd/spreka/cellbody/correctInput/masks/';
    clusterDir='/new_hdd/spreka/cellbody/correctInput/cluster/';
    styleTransTrainDir='/new_hdd/spreka/cellbody/correctInput/cluster_output_data/styleLearn_gt/';
    splitOptionsFile='/new_hdd/spreka/clustering/inputs/clustering/basicOptions_02.csv';
    % number of mask images to generate
    masks2generate=1000;
    % object type to simulate
    % objType=['nuclei','cytoplasms'];
    objType='nuclei';
elseif nargin<6
    % number of mask images to generate
    masks2generate=1000;
    % object type to simulate
    objType='nuclei';
end

artificialMaskDir=styleTransTrainDir;

clusteringType ='Kmeans-correlation-Best3Cluster';
targetImageClusterDir = fullfile(clusterDir,clusteringType);
targetMaskClusterDir = fullfile(clusterDir,[clusteringType '__MASKS']);
mkdir(fullfile(targetImageClusterDir,'group_001'));
mkdir(fullfile(targetMaskClusterDir,'group_001'));

% copy original images to one cluster and copy their masks to the mask
% cluster
l=dir(fullfile(mergedImagesDir,'*.png'));
for i=1:numel(l)
    if exist(fullfile(initialSegmentation,[l(i).name(1:end-4) '.tiff']),'file')
        copyfile(fullfile(mergedImagesDir,l(i).name),fullfile(targetImageClusterDir,'group_001',l(i).name));
        copyfile(fullfile(initialSegmentation,[l(i).name(1:end-4) '.tiff']),fullfile(targetMaskClusterDir,'group_001',[l(i).name(1:end-4) '.tiff']));
    end
end
% make required dummy folders
mkdir(fullfile(targetImageClusterDir,'thrash'));
mkdir(fullfile(targetMaskClusterDir,'thrash'));

% check missing files - by me
checkInitialSegmentation(targetImageClusterDir,initialSegmentation,'png','tiff');

% this might be unnecessary
multiplySmallFodlersForClustering(targetImageClusterDir, '', 1,'png');
multiplySmallFodlersForClustering(targetMaskClusterDir, '', 1,'tiff');

% useful part of fromClustersToStyles.m
fprintf(' ***********\n Create style transfer directory\n ***********\n');   
splitToStyleTransfer(targetImageClusterDir,targetMaskClusterDir,styleTransTrainDir, splitOptionsFile);
fprintf(' ***********\n ***********\n ***********\n READY FOR GOOGLE UPLOAD OR LOCAL STYLE TRAINING\n ***********\n ***********\n ***********\n');

% Generate masks
generateMasksToSplittedClusters_customNoMasks(styleTransTrainDir,artificialMaskDir,masks2generate,objType);

end