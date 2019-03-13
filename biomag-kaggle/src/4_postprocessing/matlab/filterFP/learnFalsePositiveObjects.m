% script for automatic filtering of false positive objects

% directory of ground truth data
gtDir = 'd:\Projects\Data Science Bowl 2018\data\__ground-truth\out_stage1_test_gold_mergedMaskLabelledImage\';
% directory of raw images 
rawImageDir = 'd:\Projects\Data Science Bowl 2018\data\__raw-images\out_stage1_test_simpleCollect\';
% directory of segmentation results
segmDir = 'd:\Projects\Data Science Bowl 2018\data\__segmentation\20180403_fulldata_coco_1x_1024_UNetcorrected_dil1px\';

saltDir = 'D:\BIOMAG\advanced-cell-classifier\LIBS\sac';
addpath(genpath(saltDir));

% featuresMap is a map where the keys are the image names and the values are
% the composite feature vectors of segmented objects
featuresMap = buildFeatureDatabase(rawImageDir, segmDir);

% annotation is a map where the keys are the image names and the values are
% the false positive values for each object in the image
initialAnnotation = annotateFalsePositives(gtDir, segmDir);

[features, labels] = convertFeatureInstances(featuresMap, initialAnnotation);

% add all FPs to training
selectedFeatures = features(labels>0,:);
selectedLabels = labels(labels>0);
selectedLabelsIdx = find(labels>0);

% add some TPs to training
leftLabelsIdx = setdiff(1:size(features,1), selectedLabelsIdx);
sampleTPIdx = leftLabelsIdx(randperm(length(leftLabelsIdx),sum(labels)));
selectedFeatures = [selectedFeatures; features(sampleTPIdx,:)];
selectedLabels = [selectedLabels; zeros(length(sampleTPIdx),1)];
selectedLabelsIdx = [selectedLabelsIdx; sampleTPIdx'];

tree = fitctree(selectedFeatures, selectedLabels);
cvmodel = crossval(tree);
L = kfoldLoss(cvmodel);
fprintf('Initial kfoldLoss: %0.3\n', L);

iter = 0;
while iter < 100
    
    % predict on unseen TP objects
    leftLabelsIdx = setdiff(1:size(features,1), selectedLabelsIdx);
    sampleTPIdx = leftLabelsIdx(randperm(length(leftLabelsIdx),sum(labels)));
    featuresToPredict = features(sampleTPIdx,:);
    realLabels = labels(sampleTPIdx);
    
    [outLabels,scores] = predict(tree, featuresToPredict);
    
    % collect wrong FP objects = TP objects getting FP label
    
    idxToAddTrain = sampleTPIdx(outLabels>0);
    selectedFeatures = [selectedFeatures; features(idxToAddTrain,:)];
    selectedLabels = [selectedLabels; zeros(length(idxToAddTrain),1)];
    selectedLabelsIdx = [selectedLabelsIdx; idxToAddTrain'];
    
    tree = fitctree(selectedFeatures, selectedLabels);
    cvmodel = crossval(tree);
    L = kfoldLoss(cvmodel);
    
    iter = iter + 1;
    
    fprintf('Iteration %03d, kfoldLoss: %0.3f (train size: %03d)\n', iter, L, length(selectedLabels));
end
