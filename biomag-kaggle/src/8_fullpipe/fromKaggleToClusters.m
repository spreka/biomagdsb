function done=fromKaggleToClusters(mergedImagesDir,clusterDir,clusteringType,initialSegmentation,sacFolder,failCounter,canContinue)
%The full pipeline from kaggle data to image clusters
%kaggleStartMatlab must be called ahead

%% Settings
%originalDir = 'e:\kaggle\KAGGLE_INPUT';
%mergedImagesDir = 'E:\kaggle\kaggle-workflow\mergedImages';
%clusterDir = 'E:\kaggle\kaggle-workflow\clusters';
%clusteringType ='Kmeans-correlation-Best3Cluster';

%% Preprocess
%{ 
fprintf(' ***********\n Preprocess\n ***********\n');
%convert data to merged input
generateDataFromDSBInput(originalDir,'outputDataType','simpleCollect');

%get the correct mergedImagesDir
splitted = strsplit(originalDir,filesep);
sourceFolder = fullfile(originalDir,['out_' splitted{end} '_simpleCollect']);
d = dir(sourceFolder);
if ~exist(mergedImagesDir,'dir'), mkdir(mergedImagesDir); end
for i=3:numel(d)
    movefile(fullfile(sourceFolder,d(i).name),fullfile(mergedImagesDir,d(i).name));
end
rmdir(sourceFolder,'s');
%}


%% Clustering 
fprintf(' ***********\n Clustering\n ***********\n');
load config.mat;
load(pretrainedDistanceLearner)
pause(2);

fprintf('Predict distances (measuring features and predict classifier)\n');
% handle sac data folder error
try
	[pairwiseDistanceMatrix,imageNames] = predictDistance(CH,mergedImagesDir);
catch ex
	fprintf('%s\n',ex.message);
	done=false;
	return;
	%rethrow(ex);
end


targetImageClusterDir = fullfile(clusterDir,clusteringType);

clusterCsvFile = clusterImages(pairwiseDistanceMatrix,imageNames,clusterDir,clusteringType);
if exist(targetImageClusterDir,'dir'), rmdir(targetImageClusterDir,'s'); end
%copy all clusters to the proper folder structure
sortToFolders(mergedImagesDir,targetImageClusterDir,clusterCsvFile);

% check if all input images have corresponding initial segmentations
checkInitialSegmentation(targetImageClusterDir,initialSegmentation,'png','tiff');

%extending clusters with single images with patching
multiplySmallFodlersForClustering(targetImageClusterDir, '', 1,'png');

%% WAIT FOR MASK RCNN output

disp('Please wait until maskRCNN results are created.');
canContinue=false;
done=true;
