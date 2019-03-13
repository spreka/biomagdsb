function findOptimalPostprocessParameters(smallScalePredictionFolder, bigScalePredictionFolder, probmapsFolder, gtFolder, parametersFile)
%
% Example1:
%   smallScalePredictionFolder = 'd:\Projects\Data Science Bowl 2018\data\contest\20180409_test\2x_2x\stage1_test\';
%   bigScalePredictionFolder = 'd:\Projects\Data Science Bowl 2018\data\contest\20180409_test\4x_2x\stage1_test\';
%   probmapsFolder = 'd:\Projects\Data Science Bowl 2018\data\contest\20180409_test\probmaps\ensembled\';
%   gtFolder = 'd:\Projects\Data Science Bowl 2018\data\__ground-truth\out_stage1_test_gold_mergedMaskLabelledImage\';
%   parametersFile = 'd:\Projects\Data Science Bowl 2018\data\contest\20180409_test\optParams.txt';
%   findOptimalPostprocessParameters(smallScalePredictionFolder, bigScalePredictionFolder, probmapsFolder, gtFolder, parametersFile);


% settings
conn = 8;
% resizeFactor = 0.5;

smallScaleImagesFolder.scale = 1;
smallScaleImagesFolder.name = smallScalePredictionFolder;

bigScaleImagesFolder.scale = 2;
bigScaleImagesFolder.name = bigScalePredictionFolder;

gtMap = readSegmentation(gtFolder, '.png');
allKeys = gtMap.keys();

smallScaleImagesMap = readSegmentation(smallScaleImagesFolder.name,'.tiff');
bigScaleImagesMap = readSegmentation(bigScaleImagesFolder.name,'.tiff');

sumProbMap = readSegmentation(probmapsFolder, '.png');

%postprocess all segmentation files: fill holes and merge touching ones
for ik=1:length(allKeys)
    smallScaleImagesMap(allKeys{ik}) = removeObjectWithinObject(smallScaleImagesMap(allKeys{ik}));
    smallScaleImagesMap(allKeys{ik}) = mergeTouchingObjects(smallScaleImagesMap(allKeys{ik}), conn);
    
    bigScaleImagesMap(allKeys{ik}) = removeObjectWithinObject(bigScaleImagesMap(allKeys{ik}));
    bigScaleImagesMap(allKeys{ik}) = mergeTouchingObjects(bigScaleImagesMap(allKeys{ik}), conn);
end 

% FMINSEARCH
% x0 = double([0.8 12000 1 1 40 0.65]);
% [x,fval] = fminsearch(@(x) optimizerFunc(x, gtMap, smallScaleImagesMap, bigScaleImagesMap, sumProbMap), x0, optimset('MaxIter',10));

% GA
x0 = [ 80, 12000, 1, 1, 35, 650]; % integer representation of parameter vector [0.8 12000 1 1 35 0.65]
lb = [ 10,     0, 1, 1, 30, 500];
ub = [100, 65535, 3, 3, 50, 800];
% Int constraint for all variable
IntCon = 1:6;
% Set options and set the start values
options = optimoptions('ga', 'InitialPopulationMatrix', x0, 'MaxTime', 60);
[x,fval,exitflag] = ga(@(x) optimizerFunc(x, gtMap, smallScaleImagesMap, bigScaleImagesMap, sumProbMap), 6, [], [] ,[],[], lb, ub, [], IntCon, options);

fprintf('Optimal parameters: overlapThresh:%0.2f, probThresh:%d, er:%d, dil:%d, minSize:%d, minOverlap:%0.2f. Score: %0.3f\n', x, 1.0-fval);

dlmwrite(parametersFile, x);
