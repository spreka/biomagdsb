%%The full pipeline from image clusters and mask rcnn output to 

%% Settings
%Please keep in mind to use the same folders as before to get it work
%properly.
%originalDir = 'E:\kaggle\kaggle-workflow\originalData';
%mergedImagesDir = 'E:\kaggle\kaggle-workflow\mergedImages';
%clusterDir = 'E:\kaggle\kaggle-workflow\clusters';
%initialSegmentation = 'E:\kaggle\kaggle-workflow\initSegmentation';
%styleTransTrainDir = 'E:\kaggle\kaggle-workflow\StyleLearnInput';
%artificialMaskDir = 'E:\kaggle\kaggle-workflow\syntheticMasks';

%splitOptionsFile = 'E:\kaggle\Code\src\3_Clustering\splitOptions\basicOptions_32.csv';
clusteringType ='Kmeans-correlation-Best3Cluster';

%% Augment sinlge image clusters
fprintf(' ***********\n Augment sinlge image clusters\n ***********\n');
clusterCsvFile = fullfile(clusterDir,['predictedStyles_' clusteringType '.csv']);
targetMaskClusterDir = fullfile(clusterDir,[clusteringType '__MASKS']);

sortToFolders(initialSegmentation,targetMaskClusterDir,clusterCsvFile); 

multiplySmallFodlersForClustering(targetMaskClusterDir, '', 1,'tiff');  % was png

%% Create style transfer directory
fprintf(' ***********\n Create style transfer directory\n ***********\n');
targetImageClusterDir = fullfile(clusterDir,clusteringType);    
splitToStyleTransfer(targetImageClusterDir,targetMaskClusterDir,styleTransTrainDir, splitOptionsFile);

fprintf(' ***********\n ***********\n ***********\n READY FOR GOOGLE UPLOAD OR LOCAL STYLE TRAINING\n ***********\n ***********\n ***********\n');

%% Generate masks
% objType=['nuclei','cytoplasms'];
objType='nuclei';
generateMasksToSplittedClusters(styleTransTrainDir,artificialMaskDir,objType);