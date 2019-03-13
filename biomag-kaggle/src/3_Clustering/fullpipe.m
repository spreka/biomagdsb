%fullPipeline

%clusterWorkDirectory = 'D:\Ábel\SZBK\Projects\Kaggle\Abel\Clustering';
load config.mat;
trainDir = fullfile(clusterWorkDirectory,'train');
trainStyleCsv = fullfile(clusterWorkDirectory,'stylesToTrain.csv');
testDir = fullfile(clusterWorkDirectory,'test');
csvFilePath = clusterWorkDirectory;
targetDir = fullfile(clusterWorkDirectory,'clusters');

% {
tic
CH = trainDistance(trainDir,trainStyleCsv);
toc
% {
tic
[DD,imageNames] = predictDistance(CH,testDir);
toc
%}
load(fullfile(clusterWorkDirectory,'SavedDistanceMatrix.mat'));
%save(fullfile(clusterWorkDirectory,'SavedDistanceMatrix.mat'));

%type = 'Kmeans-cosine-Best5Cluster';
type ='Kmeans-correlation-Best3Cluster';


targetDir = fullfile(targetDir,type);
csvFile = clusterImages(DD,imageNames,csvFilePath,type);
copyfile(csvFile,[csvFile(1:end-4) '_Original.csv']);
if exist(targetDir,'dir'), rmdir(targetDir,'s'); end
sortToFolders(testDir,targetDir,csvFile);
%}

