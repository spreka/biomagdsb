function CH = trainDistance(trainDir,styleCsv)
%gives back a trained CommonHandles to use for predictions (it is a cellarray to keep the possibility to build an ensemble)

%wekaNames: cellarray where each entry is either a number for the selected
%method or the weka classifier name.
%1: MLP
%2: RandomForest

wekaNames = {1,2,3};
nofModels = length(wekaNames);

for i=1:nofModels
    if isnumeric(wekaNames{i})
        list = sacGetSupportedList('classifiers');
        wekaNames{i} = list{wekaNames{i}};               
    end
end

%train directory
if nargin<1
    trainDir = 'd:\Ábel\SZBK\Projects\Kaggle\Abel\Clustering\train\';
end
if nargin<2
styleCsv = 'd:\Ábel\SZBK\Projects\Kaggle\Abel\Clustering\stylesToTrain.csv';
end
nofNonSimilarPairsInTrainingSet = 1000;

T = readtable(styleCsv);

nofClusters = max(T.Style);

[features,~,imageNames] = loadBasicFeatures(trainDir,0,1);
featureMap = containers.Map;
for i=1:length(imageNames)
    featureMap(imageNames{i}) = features{i};
end

trainFeatures = cell(length(features)^2,1);
fakeImageNames = cell(length(features)^2,1);
similar = containers.Map;

%put in the similar images
counter = 1;
for i=1:nofClusters
    imgsInSameCluster = T.Name(T.Style == i);
    for j=1:numel(imgsInSameCluster)-1
        for k=j+1:numel(imgsInSameCluster)
            trainFeatures{counter} = diffFeature(featureMap(imgsInSameCluster{j}),featureMap(imgsInSameCluster{k}));
            fakeImageNames{counter} = ['FakeImg' num2str(counter,'%04d')];
            similar(fakeImageNames{counter}) = [1 0];
            counter = counter+1;
        end
    end
end

%put in the non similar images
for i=1:nofNonSimilarPairsInTrainingSet
    a=randi(nofClusters);
    b=randi(nofClusters);
    
    if a~=b
        aImgs = T.Name(T.Style == a);
        bImgs = T.Name(T.Style == b);
        ai=randi(numel(aImgs));
        bi=randi(numel(bImgs));        
        trainFeatures{counter} = diffFeature(featureMap(aImgs{ai}),featureMap(bImgs{bi}));
        fakeImageNames{counter} = ['FakeImg' num2str(counter,'%04d')];
        similar(fakeImageNames{counter}) = [0 1];
        counter = counter + 1;
    end
end

trainFeatures(counter:end) = [];
fakeImageNames(counter:end) = [];

CHBasic = mapFeaturesToCommonHandles(trainFeatures,similar,fakeImageNames);
CH = cell(1,nofModels);

for i=1:nofModels
    CH{i} = CHBasic;
    CH{i}.ClassifierNames = wekaNames(i);
    CH{i}.SelectedClassifier = 1;
    try
        CH{i} = trainClassifier(CH{i});
    catch
        load config.mat;
        sacInit(fullfile(codeBase,'1_metalearning','matlab','sac'));
        CH{i} = trainClassifier(CH{i});
    end

end

