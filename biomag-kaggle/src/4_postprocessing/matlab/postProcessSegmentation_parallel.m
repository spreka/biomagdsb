function postProcessSegmentation_parallel( smallScaleImagesFolder, bigScaleImagesFolder, probMapsFolder, outFolder, parameters)
% postProcessSegmentation
%
% Example1:
%
%   smallScaleImagesFolder = struct('name', 'd:\Projects\Data Science Bowl 2018\data\contest\20180409_test\2x_2x\stage1_test\','ext','.tiff','scale',1);
%   bigScaleImagesFolder = struct('name', 'd:\Projects\Data Science Bowl 2018\data\contest\20180409_test\4x_2x\stage1_test\','ext','.tiff','scale',2);
%   probMapsFolder = struct('name', 'd:\Projects\Data Science Bowl 2018\data\contest\20180409_test\probmaps\ensembled\','ext','.png');
%   outFolder = struct('name', 'd:\Projects\Data Science Bowl 2018\data\contest\20180409_test\outTestPetersParams','ext','.tiff');
%   parameters = struct('minSize',25, 'overlapThresh',0.67, 'probThresh',9547,'erosionRadius',1,'dilationRadius',1,'conn',8,'minOverlap',0.77,'maxVParam',31,'cAreaParam',22);
%   postProcessSegmentation_parallel( smallScaleImagesFolder, bigScaleImagesFolder, probMapsFolder, outFolder, parameters)

% resizeFactor = 0.5;

if ~isstruct(parameters)
    error('kaggle:postprocessing:invalidInputError','Missing last argument parameter struct.');
end

if ~isfield(parameters, 'minSize')
    parameters.minSize = 40; % magic tested number from Peter
    warning('No ''minSize'' field exist for parameters. It is set to default value (40).');
end

if ~isfield(parameters, 'overlapThresh')
    parameters.overlapThresh = 0.8; % magic tested number from Peter
    warning('No ''overlapThresh'' field exist for parameters, which controls the merging algorithm. It is set to default value (0.8).');
end

if ~isfield(parameters, 'minOverlap')
    parameters.minOverlap = 0.65; % magic tested number from Peter
    warning('No ''minOverlap'' field exist for parameters, which controls the acceptance of FP objects. It is set to default value (0.65).');
end

if ~isfield(parameters, 'probThresh')
    parameters.probThresh = 8656; % magic tested number from Lassi
    warning('No ''probThresh'' field exist for parameters. It is set to default value (8656).');
end

if ~isfield(parameters, 'erosionRadius')
    parameters.erosionRadius = 1; % magic tested number from Peter & Reka
    warning('No ''erosionRadius'' field exist for parameters. It is set to default value (1).');
end

if ~isfield(parameters, 'dilationRadius')
    parameters.dilationRadius = 1; % magic tested number from Peter & Reka
    warning('No ''dilationRadius'' field exist for parameters. It is set to default value (1).');
end

if ~isfield(parameters, 'maxVParam')
    parameters.maxVParam = 31; % optimized parameter from Norbi
    warning('No ''maxVParam'' field exist for parameters. It is set to default value (31).');
end

if ~isfield(parameters, 'cAreaParam')
    parameters.cAreaParam = 22; % optimized parameter from Norbi
    warning('No ''cAreaParam'' field exist for parameters. It is set to default value (22).');
end

if ~isfield(parameters, 'conn')
    parameters.conn = 8; % magic tested number from Peter
    warning('No ''conn'' field exist for parameters. It is set to default value (8).');
end

smallScaleImagesMap = readSegmentation(smallScaleImagesFolder.name, smallScaleImagesFolder.ext);
bigScaleImagesMap = readSegmentation(bigScaleImagesFolder.name, bigScaleImagesFolder.ext);

allSmallKeys = smallScaleImagesMap.keys();
allBigKeys = bigScaleImagesMap.keys();

sumProbMap = readSegmentation(probMapsFolder.name, probMapsFolder.ext);
probMapKeys = sumProbMap.keys();

allKeys = unique([allSmallKeys{:} allBigKeys probMapKeys]);
% TODO make it more elegant
% for ik=1:length(allKeys)
%     sumProbMap(allKeys{ik}) = double(sumProbMap(allKeys{ik}))/2^16;
% end

%postprocess all segmentations: fill holes and merge touching ones
smallScaleImagesCellArray = cell(length(allKeys), 1);
bigScaleImagesCellArray = cell(length(allKeys), 1);
% probMapsCellArray = cell(length(allKeys), 1);

for ik=1:length(allKeys)
    if ~isKey(smallScaleImagesMap, allKeys{ik})
        smallScaleImagesMap(allKeys{ik}) = uint16(zeros(size(sumProbMap(allKeys{ik}),1),size(sumProbMap(allKeys{ik}),2)));
    end
    if ~isKey(bigScaleImagesMap, allKeys{ik})
        bigScaleImagesMap(allKeys{ik}) = uint16(zeros(size(sumProbMap(allKeys{ik}),1),size(sumProbMap(allKeys{ik}),2)));
    end
end

fprintf('Filling holes (& clearing embedded objects)...\n');
parfor ik=1:length(allKeys)
	smallScaleImagesCellArray{ik} = removeObjectWithinObject(smallScaleImagesMap(allKeys{ik}));
    bigScaleImagesCellArray{ik} = removeObjectWithinObject(bigScaleImagesMap(allKeys{ik}));    
end

fprintf('Merging touching concave objects...\n');
parfor ik=1:length(allKeys)
    smallScaleImagesCellArray2{ik} = mergeTouchingObjects(smallScaleImagesCellArray{ik}, parameters.conn);
    bigScaleImagesCellArray2{ik} = mergeTouchingObjects(bigScaleImagesCellArray{ik}, parameters.conn);
end

% discard small objects 1st round
fprintf('Discarding small objects...\n');
parfor ik=1:length(allKeys)
    smallScaleImagesCellArray2{ik} = removeSmallObjects(smallScaleImagesCellArray{ik}, parameters.minSize);
    bigScaleImagesCellArray2{ik} = removeSmallObjects(bigScaleImagesCellArray{ik}, parameters.minSize);
end

for ik=1:length(allKeys)
    smallScaleImagesMap(allKeys{ik}) = imresize(smallScaleImagesCellArray2{ik}, size(sumProbMap(allKeys{ik})), 'nearest');
    bigScaleImagesMap(allKeys{ik}) = imresize(bigScaleImagesCellArray2{ik}, size(sumProbMap(allKeys{ik})), 'nearest');
end


% writeSegmentation(smallScaleImagesMap, 'd:\kaggle\temp\1x', '.tiff');
% writeSegmentation(bigScaleImagesMap, 'd:\kaggle\temp\2x', '.tiff');

% mergedImagesMap = mergeScalesMap2(struct('map',smallScaleImagesMap,'scale',smallScaleImagesFolder.scale),...
%                                   struct('map',bigScaleImagesMap,'scale',bigScaleImagesFolder.scale),...
%                                   parameters.scaleThresh);
                              
mergedImagesMap = mergeScalesMapByDropping2(struct('map',smallScaleImagesMap,'scale',smallScaleImagesFolder.scale),...
                                  struct('map',bigScaleImagesMap,'scale',bigScaleImagesFolder.scale),...
                                  parameters.overlapThresh);
                              
% writeSegmentation(mergedImagesMap, 'd:\kaggle\temp\master', '.tiff');

postProcessedImgMap = correctWithUnet2(mergedImagesMap, sumProbMap, parameters.probThresh, parameters.erosionRadius, parameters.dilationRadius, parameters.minOverlap, parameters.maxVParam, parameters.cAreaParam);
% writeSegmentation(postProcessedImgMap, 'd:\kaggle\temp\masterBlaster', '.tiff');

writeSegmentation(postProcessedImgMap, outFolder.name, outFolder.ext);
