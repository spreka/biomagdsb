%% postprocess script
clear;
gtFolder = 'd:\Projects\Data Science Bowl 2018\data\__ground-truth\out_stage1_test_gold_mergedMaskLabelledImage\';

inFolder = 'd:\Projects\Data Science Bowl 2018\data\contest\20180409\';
inProbmapFolders = {'d:\Projects\Data Science Bowl 2018\data\contest\20180409\probmaps\ensembled\'};
outFolder = 'd:\Projects\Data Science Bowl 2018\data\contest\20180409\';
mkdir(outFolder);

scalesFolders(1).scale = 1;
scalesFolders(1).name = fullfile(inFolder, '2x_2x', 'stage1_test');

scalesFolders(2).scale = 2;
scalesFolders(2).name = fullfile(inFolder, '4x_2x', 'stage1_test');

resultsFile = 'd:\Projects\Data Science Bowl 2018\data\contest\20180409\paramsearchresult.txt';
fid = fopen(resultsFile, 'w');

%% settings
conn = 8;
% TODO test param
erosionRadii = [0:4];
dilationRadii = [0:4];
minSizes = [5:5:75];
scaleThreshs = [10:30];
% probThresholds = 8656;
probThresholds = 8656-500:100:8656+500;
% resize working image to original size
resizeFactor = 0.5;

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

% probMaps = cell(1, length(inProbmapFolders));
sumProbMap = containers.Map;
for i=1:length(allKeys)
    sumProbMap(allKeys{i}) = zeros(size(gtMap(allKeys{i})));
end

for i=1:length(inProbmapFolders)
    probMaps = readSegmentation(inProbmapFolders{i}, '.png');
    for ik=1:length(allKeys)
        sumProbMap(allKeys{ik}) = sumProbMap(allKeys{ik}) + double(probMaps(allKeys{ik}));
    end
    sumProbMap(allKeys{ik}) = sumProbMap(allKeys{ik})/2^16;
end

tic

smallScaleImagesMap = readSegmentation(scalesFolders(1).name,'.tiff');
bigScaleImagesMap = readSegmentation(scalesFolders(2).name,'.tiff');

%postprocess all segmentation files: fill holes and merge touchin ones
for ik=1:length(allKeys)
    smallScaleImagesMap(allKeys{ik}) = removeObjectWithinObject(smallScaleImagesMap(allKeys{ik}));
    smallScaleImagesMap(allKeys{ik}) = mergeTouchingObjects(smallScaleImagesMap(allKeys{ik}), conn);
    
    bigScaleImagesMap(allKeys{ik}) = removeObjectWithinObject(smallScaleImagesMap(allKeys{ik}));
    bigScaleImagesMap(allKeys{ik}) = mergeTouchingObjects(bigScaleImagesMap(allKeys{ik}), conn);
end 

for scaleThrsh = scaleThreshs
    for probThresh = probThresholds
        for erosionRadius = erosionRadii
            for dilationRadius = dilationRadii
                for minSize = minSizes
                    fprintf('Calc corrected segmentations (scaleTh:%d, er:%dpx, dil:%dpx, min:%d, probTh:%d)\n',scaleThrsh,erosionRadius,dilationRadius,minSize,probThresh);
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

toc

fclose(fid);

delete(gcp('nocreate'));
