function postProcessedImgMap = postProcessSegmentation2( smallScaleImagesMapStruct, bigScaleImagesMapStruct, sumProbMap, parameters)
% postProcessSegmentation
%

if ~isstruct(parameters)
    error('kaggle:postprocessing:invalidInputError','Missing last argument parameter struct.');
end

if ~isfield(parameters, 'minSize')
    parameters.minSize = 40; % magic tested number from Peter
    warning('No ''minSize'' field exist for parameters. It is set to default value (40).');
end

if ~isfield(parameters, 'scaleThresh')
    parameters.scaleThresh = 14; % magic tested number from Peter
    warning('No ''scaleThresh'' field exist for parameters. It is set to default value (14).');
end

if ~isfield(parameters, 'probThresh')
    parameters.probThresh = 8656; % magic tested number from Lassi
    warning('No ''probThresh'' field exist for parameters. It is set to default value (8656).');
end

if ~isfield(parameters, 'erosionRadius')
    parameters.erosionRadius = 0; % magic tested number from Peter
    warning('No ''erosionRadius'' field exist for parameters. It is set to default value (0).');
end

if ~isfield(parameters, 'dilationRadius')
    parameters.dilationRadius = 2; % magic tested number from Peter
    warning('No ''dilationRadius'' field exist for parameters. It is set to default value (2).');
end

if ~isfield(parameters, 'conn')
    parameters.conn = 4; % magic tested number from Peter
    warning('No ''conn'' field exist for parameters. It is set to default value (4).');
end

% smallScaleImagesMap = readSegmentation(smallScaleImagesFolder.name, smallScaleImagesFolder.ext);
% bigScaleImagesMap = readSegmentation(bigScaleImagesFolder.name, bigScaleImagesFolder.ext);

smallScaleImagesMap = smallScaleImagesMapStruct.map;
bigScaleImagesMap = bigScaleImagesMapStruct.map;


allKeys = smallScaleImagesMap.keys();

%postprocess all segmentations: fill holes and merge touching ones
% for ik=1:length(allKeys)
%     smallScaleImagesMap(allKeys{ik}) = removeObjectWithinObject(smallScaleImagesMap(allKeys{ik}));
%     smallScaleImagesMap(allKeys{ik}) = mergeTouchingObjects(smallScaleImagesMap(allKeys{ik}), parameters.conn);
%     
%     bigScaleImagesMap(allKeys{ik}) = removeObjectWithinObject(smallScaleImagesMap(allKeys{ik}));
%     bigScaleImagesMap(allKeys{ik}) = mergeTouchingObjects(bigScaleImagesMap(allKeys{ik}), parameters.conn);
% end

% discard small objects 1st round
for i=1:length(allKeys)
    smallScaleImage = removeSmallObjects(smallScaleImagesMap(allKeys{i}), parameters.minSize);
    smallScaleImagesMap(allKeys{i}) = imresize(smallScaleImage, size(sumProbMap(allKeys{i})), 'nearest');
    bigScaleImage = removeSmallObjects(bigScaleImagesMap(allKeys{i}), parameters.minSize);
    bigScaleImagesMap(allKeys{i}) = imresize(bigScaleImage, size(sumProbMap(allKeys{i})), 'nearest');
end

mergedImagesMap = mergeScalesMap2(struct('map',smallScaleImagesMap,'scale',smallScaleImagesMapStruct.scale),...
                                  struct('map',bigScaleImagesMap,'scale',bigScaleImagesMapStruct.scale),...
                                  parameters.scaleThresh);

postProcessedImgMap = correctWithUnet2(mergedImagesMap, sumProbMap, parameters.probThresh, parameters.erosionRadius, parameters.dilationRadius);
