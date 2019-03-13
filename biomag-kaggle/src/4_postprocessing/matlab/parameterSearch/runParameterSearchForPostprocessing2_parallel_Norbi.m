%% postprocess script
% clear;

% gtFolder = '/media/baran/LinuxData/Downloads/Challange/Optimizer/kaggle_data_21080411/groundtruth2/'
% inFolder = '/media/baran/LinuxData/Downloads/Challange/Optimizer/kaggle_data_21080411/'
% inProbmapFolders = {'/media/baran/LinuxData/Downloads/Challange/Optimizer/kaggle_data_21080411/probmaps/ensembled/'};
% outFolder = '/media/baran/LinuxData/Downloads/Challange/Optimizer/kaggle_data_21080411/results/';

gtFolder = 'd:\Projects\Data Science Bowl 2018\data\__ground-truth\out_stage1_test_gold_mergedMaskLabelledImage\';

inFolder = 'd:\Projects\Data Science Bowl 2018\data\contest\20180409_test\';
inProbmapFolders = {'d:\Projects\Data Science Bowl 2018\data\contest\20180409_test\probmaps\ensembled\'};
outFolder = 'd:\Projects\Data Science Bowl 2018\data\contest\20180409_test\';

mkdir(outFolder);

smallScaleImagesFolder.scale = 1;
smallScaleImagesFolder.name = fullfile(inFolder, '2x_2x', 'stage1_test');

bigScaleImagesFolder.scale = 2;
bigScaleImagesFolder.name = fullfile(inFolder, '4x_2x', 'stage1_test');

resultsFile = '/media/baran/LinuxData/Downloads/Challange/Optimizer/kaggle_data_21080411/results/paramsearchresult.txt';
fid = fopen(resultsFile, 'w');


%% settings
conn = 8;

%{
% TODO test param
erosionRadii = [0:3]
dilationRadii = [0:3]
minSizes = [5:10:75]
scaleThreshs = [10:2:30];
probThresholds = [8656-500:100:8656+500];
% resize working image to original size
resizeFactor = 0.5;

parameterArray(numel(erosionRadii)*numel(dilationRadii)*numel(minSizes)*numel(scaleThreshs)*numel(probThresholds)) = struct('id',0, 'parameters', struct('minSize',0, 'scaleThresh',0,'probThresh',0,'erosionRadius',0,'dilationRadius',0));

ind = 0;
for minSizeInd = 1:length(minSizes)
    for scaleThrshInd = 1:length(scaleThreshs)
        for probThreshInd = 1:length(probThresholds)
            for erosionRadiusInd = 1:length(erosionRadii)
                for dilationRadiusInd = 1:length(dilationRadii)
                    ind = ind+1;
                    parameterArray(ind) = struct('id',ind, 'parameters', struct('minSize',minSizes(minSizeInd), 'scaleThresh',scaleThreshs(scaleThrshInd),'probThresh',probThresholds(probThreshInd),'erosionRadius',erosionRadii(erosionRadiusInd),'dilationRadius',dilationRadii(dilationRadiusInd)));
                end
            end
        end
    end
end

resStringCellArray = cell(numel(parameterArray),1);
%}

saveMultiScaleImages = 0;
saveAllParamsResults = 0;
evalAllResults = 1;

p = gcp('nocreate');
if isempty(p)
    parpool('local',feature('numcores'));
end
% delete(gcp('nocreate'));


%% making folder structure

if saveMultiScaleImages
    masterScaleFolder = fullfile(outFolder, 'masterScale');
    mkdir(masterScaleFolder);
end

% TODO change it for all parameters
if saveAllParamsResults
    for scaleThrsh = scaleThreshs
        for erosionRadius = erosionRadii
            for dilationRadius = dilationRadii
                for minSize = minSizes
                    outFolderCaseName = sprintf('masterScale_e%d_d%d_min%d', erosionRadius, dilationRadius, minSize);
                    outFolderCaseFullPath = fullfile(outFolder,outFolderCaseName);
                    mkdir(outFolderCaseFullPath);
                end
            end
        end
    end
end

gtMap = readSegmentation(gtFolder, '.png');
allKeys = gtMap.keys();

sumProbMap = readSegmentation(inProbmapFolders{1}, '.png');

tic

smallScaleImagesMap = readSegmentation(smallScaleImagesFolder.name,'.tiff');
bigScaleImagesMap = readSegmentation(bigScaleImagesFolder.name,'.tiff');

%postprocess all segmentation files: fill holes and merge touching ones
%
for ik=1:length(allKeys)
    smallScaleImagesMap(allKeys{ik}) = removeObjectWithinObject(smallScaleImagesMap(allKeys{ik}));
    smallScaleImagesMap(allKeys{ik}) = mergeTouchingObjects(smallScaleImagesMap(allKeys{ik}), conn);
    
    bigScaleImagesMap(allKeys{ik}) = removeObjectWithinObject(bigScaleImagesMap(allKeys{ik}));
    bigScaleImagesMap(allKeys{ik}) = mergeTouchingObjects(bigScaleImagesMap(allKeys{ik}), conn);
end
%


%{
% main loop
parfor i=1:length(parameterArray)
    parameters = parameterArray(i).parameters;
    postProcessedMap = postProcessSegmentation2( struct('map', smallScaleImagesMap, 'scale', smallScaleImagesFolder.scale), struct('map', bigScaleImagesMap, 'scale', bigScaleImagesFolder.scale), sumProbMap, parameters);
    if evalAllResults
        score = evaluation2(gtMap, postProcessedMap);
%         fprintf('Score for corrected segmentations (scaleTh:%d, er:%dpx, dil:%dpx, min:%d, probTh:%d): %0.3f\n',...
%                 scaleThresh,erosionRadius,dilationRadius,minSize,probThresh,mean(cell2mat(score.values)));
%         fidCurrent = fopen(fullfile(outFolder,sprintf('%06d_score.txt',parameterArray(i).id)),'w'); 
%         fprintf(fidCurrent,'Score for corrected segmentations (scaleTh:%d, er:%dpx, dil:%dpx, min:%d, probTh:%d): %0.3f\n',...
%                 parameters.scaleThresh,parameters.erosionRadius,parameters.dilationRadius,parameters.minSize,parameters.probThresh,mean(cell2mat(score.values)));
%         fclose(fidCurrent);
        resStringCellArray{i} = sprintf('Score for corrected segmentations (scaleTh:%d, er:%dpx, dil:%dpx, min:%d, probTh:%d): %0.3f\n',...
                         parameters.scaleThresh,parameters.erosionRadius,parameters.dilationRadius,parameters.minSize,parameters.probThresh,mean(cell2mat(score.values)));
    end
end

for i=1:length(resStringCellArray)
    fprintf(fid,[resStringCellArray{i}]);
end
%}

toc

%{

for minSize = minSizes
    % discard small objects 1st round
    for i=1:length(allKeys)
        smallScaleImage = removeSmallObjects(smallScaleImagesMap(allKeys{i}), minSize);
        smallScaleImagesMap(allKeys{i}) = imresize(smallScaleImage, size(sumProbMap(allKeys{i})), 'nearest');
        bigScaleImage = removeSmallObjects(bigScaleImagesMap(allKeys{i}), minSize);
        bigScaleImagesMap(allKeys{i}) = imresize(bigScaleImage, size(sumProbMap(allKeys{i})), 'nearest');
    end
    for scaleThrsh = scaleThreshs
        % merging images from 2 folder
        mergedImgMap = mergeScalesMap2(struct('map',smallScaleImagesMap, 'scale', 1), struct('map',bigScaleImagesMap, 'scale', 2), scaleThrsh);
        for probThresh = probThresholds
            for erosionRadius = erosionRadii
                for dilationRadius = dilationRadii
                    
                    %                     fprintf('Calc corrected segmentations (scaleTh:%d, er:%dpx, dil:%dpx, min:%d, probTh:%d)\n',scaleThrsh,erosionRadius,dilationRadius,minSize,probThresh);
                    outFinalImageMap = mergeUnetAndAll(smallScaleImagesMap, bigScaleImagesMap, sumProbMap, resizeFactor, scaleThrsh, probThresh, erosionRadius, dilationRadius, minSize);
                    
                    if evalAllResults
                        score = evaluation2(gtMap, outFinalImageMap);
                        fprintf('Score for corrected segmentations (scaleTh:%d, er:%dpx, dil:%dpx, min:%d, probTh:%d): %0.3f\n',scaleThrsh,erosionRadius,dilationRadius,minSize,probThresh,mean(cell2mat(score.values)));
                        fprintf(fid,'Score for corrected segmentations (scaleTh:%d, er:%dpx, dil:%dpx, min:%d, probTh:%d): %0.3f\n',scaleThrsh,erosionRadius,dilationRadius,minSize,probThresh,mean(cell2mat(score.values)));
                    end
                end
            end
        end
    end
end


%}

% FMINSEARCH
x0 = double([14 12000 1 1 40]);
fminsearch(@(x)optimizerFunc(x, gtMap, smallScaleImagesMap, bigScaleImagesMap, sumProbMap, fid, resizeFactor), x0);

% FMINCON
%x0 = double([0.14 0.120 0.1 0.1 0.40]);
%lb = double([0.05 0.100 0.0 0.1 0.05]);
%ub = double([0.35 0.130 0.4 0.4 0.50]);
%opts = optimset('Display','iter','Algorithm','interior-point', 'MaxIter', inf, 'MaxFunEvals', inf);
%[x,fval,flag] = fmincon(@optimizerFunc, x0, [], [], [], [], lb, ub, [], opts);

fclose(fid);
delete(gcp('nocreate'));
