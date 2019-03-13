function postProcessSegmentation( smallScaleImagesFolder, bigScaleImagesFolder, probMapsFolder, outFolder, parameters)
% postProcessSegmentation
%
% Example1:
%
%   smallScaleImagesFolder = struct('name', 'd:\Projects\Data Science Bowl 2018\data\contest\20180409_test\2x_2x\stage1_test\','ext','.tiff','scale',1);
%   bigScaleImagesFolder = struct('name', 'd:\Projects\Data Science Bowl 2018\data\contest\20180409_test\4x_2x\stage1_test\','ext','.tiff','scale',2);
%   probMapsFolder = struct('name', 'd:\Projects\Data Science Bowl 2018\data\contest\20180409_test\probmaps\ensembled\','ext','.png');
%   outFolder = struct('name', 'd:\Projects\Data Science Bowl 2018\data\contest\20180409_test\outTestPetersParams','ext','.tiff');
%   parameters = struct('minSize',40, 'overlapThresh',0.8, 'probThresh',8656,'erosionRadius',1,'dilationRadius',1,'conn',8,'minOverlap',0.65);
%   postProcessSegmentation( smallScaleImagesFolder, bigScaleImagesFolder, probMapsFolder, outFolder, parameters)


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
    warning('No ''erosionRadius'' field exist for parameters. It is set to default value (0).');
end

if ~isfield(parameters, 'dilationRadius')
    parameters.dilationRadius = 1; % magic tested number from Peter & Reka
    warning('No ''dilationRadius'' field exist for parameters. It is set to default value (2).');
end

if ~isfield(parameters, 'conn')
    parameters.conn = 4; % magic tested number from Peter
    warning('No ''conn'' field exist for parameters. It is set to default value (4).');
end

smallScaleImagesMap = readSegmentation(smallScaleImagesFolder.name, smallScaleImagesFolder.ext);
bigScaleImagesMap = readSegmentation(bigScaleImagesFolder.name, bigScaleImagesFolder.ext);
allKeys = smallScaleImagesMap.keys();

sumProbMap = readSegmentation(probMapsFolder.name, probMapsFolder.ext);

allKeys = sumProbMap.keys();

for ik=1:length(allKeys)
    if ~isKey(smallScaleImagesMap, allKeys{ik})
        smallScaleImagesMap(allKeys{ik}) = uint16(zeros(size(sumProbMap(allKeys{ik}),1),size(sumProbMap(allKeys{ik}),2)));
    end
    if ~isKey(bigScaleImagesMap, allKeys{ik})
        bigScaleImagesMap(allKeys{ik}) = uint16(zeros(size(sumProbMap(allKeys{ik}),1),size(sumProbMap(allKeys{ik}),2)));
    end
end

%postprocess all segmentations: fill holes and merge touching ones
for ik=1:length(allKeys)
    smallScaleImagesMap(allKeys{ik}) = removeObjectWithinObject(smallScaleImagesMap(allKeys{ik}));
    smallScaleImagesMap(allKeys{ik}) = mergeTouchingObjects(smallScaleImagesMap(allKeys{ik}), parameters.conn);
    
    bigScaleImagesMap(allKeys{ik}) = removeObjectWithinObject(bigScaleImagesMap(allKeys{ik}));
    bigScaleImagesMap(allKeys{ik}) = mergeTouchingObjects(bigScaleImagesMap(allKeys{ik}), parameters.conn);
end

% discard small objects 1st round
for i=1:length(allKeys)
    smallScaleImage = removeSmallObjects(smallScaleImagesMap(allKeys{i}), parameters.minSize);
    smallScaleImagesMap(allKeys{i}) = imresize(smallScaleImage, size(sumProbMap(allKeys{i})), 'nearest');
    bigScaleImage = removeSmallObjects(bigScaleImagesMap(allKeys{i}), parameters.minSize);
    bigScaleImagesMap(allKeys{i}) = imresize(bigScaleImage, size(sumProbMap(allKeys{i})), 'nearest');
end

% mergedImagesMap = mergeScalesMap2(struct('map',smallScaleImagesMap,'scale',smallScaleImagesFolder.scale),...
%                                   struct('map',bigScaleImagesMap,'scale',bigScaleImagesFolder.scale),...
%                                   parameters.scaleThresh);
                              
mergedImagesMap = mergeScalesMapByDropping2(struct('map',smallScaleImagesMap,'scale',smallScaleImagesFolder.scale),...
                                  struct('map',bigScaleImagesMap,'scale',bigScaleImagesFolder.scale),...
                                  parameters.overlapThresh);

postProcessedImgMap = correctWithUnet2(mergedImagesMap, sumProbMap, parameters.probThresh, parameters.erosionRadius, parameters.dilationRadius, parameters.minOverlap, parameters.maxVParam, parameters.cAreaParam);

writeSegmentation(postProcessedImgMap, outFolder.name, outFolder.ext);
