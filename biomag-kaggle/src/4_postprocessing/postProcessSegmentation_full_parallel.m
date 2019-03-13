function postProcessSegmentation_full_parallel( smallScaleImagesFolder, bigScaleImagesFolder, probMapsFolder, outFolder, parameters)
% postProcessSegmentation
%
% Example1:
%
%   smallScaleImagesFolder = struct('name', 'd:\Projects\Data Science Bowl 2018\data\contest\20180409_test\2x_2x\stage1_test\','ext','.tiff','scale',1);
%   bigScaleImagesFolder = struct('name', 'd:\Projects\Data Science Bowl 2018\data\contest\20180409_test\4x_2x\stage1_test\','ext','.tiff','scale',2);
%   probMapsFolder = struct('name', 'd:\Projects\Data Science Bowl 2018\data\contest\20180409_test\probmaps\ensembled\','ext','.png');
%   outFolder = struct('name', 'd:\Projects\Data Science Bowl 2018\data\contest\20180409_test\outTestPetersParams','ext','.tiff');
%   parameters = struct('minSize',25, 'overlapThresh',0.67, 'probThresh',9547,'erosionRadius',1,'dilationRadius',1,'conn',8,'minOverlap',0.77,'maxVParam',31,'cAreaParam',22);
%   postProcessSegmentation_full_parallel( smallScaleImagesFolder, bigScaleImagesFolder, probMapsFolder, outFolder, parameters);

% resizeFactor = 0.5;

if ~exist(outFolder.name, 'dir')
    mkdir(outFolder.name);
end

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


probMapList = dir(fullfile(probMapsFolder.name, ['*' probMapsFolder.ext]));

parfor imInd = 1:numel(probMapList)
    probMapName = probMapList(imInd).name;
    [~, imageID,~] = fileparts(probMapName);
    
    % read inputs to postprocess
    probMap = imread(fullfile(probMapsFolder.name,probMapName));
    if exist(fullfile(smallScaleImagesFolder.name,[imageID smallScaleImagesFolder.ext]),'file')
        smallScaleImage = imread(fullfile(smallScaleImagesFolder.name,[imageID smallScaleImagesFolder.ext]));
    else
        smallScaleImage = uint16(zeros(size(probMap,1),size(probMap,2)));
    end
    if exist(fullfile(bigScaleImagesFolder.name, [imageID bigScaleImagesFolder.ext]),'file')
        bigScaleImage = imread(fullfile(bigScaleImagesFolder.name, [imageID bigScaleImagesFolder.ext]));
    else
        bigScaleImage = uint16(zeros(size(probMap,1),size(probMap,2)));
    end
    
    % filling holes (including discard of embedded objects)
    smallScaleImage = removeObjectWithinObject(smallScaleImage);
    bigScaleImage = removeObjectWithinObject(bigScaleImage);
    
    % merging touching objects if concavity is better
    smallScaleImage = mergeTouchingObjects(smallScaleImage,parameters.conn);
    bigScaleImage = mergeTouchingObjects(bigScaleImage,parameters.conn);
    
    % discard small objects
    smallScaleImage = removeSmallObjects(smallScaleImage, parameters.minSize);
    bigScaleImage = removeSmallObjects(bigScaleImage, parameters.minSize);
    
    % resizing to original size
    smallScaleImage = imresize(smallScaleImage, [size(probMap,1),size(probMap,2)], 'nearest');
    bigScaleImage = imresize(bigScaleImage, [size(probMap,1),size(probMap,2)], 'nearest');
    
    % merge 2 scales by  dropping
    mergedImage = mergeTwoMasksByDropping(smallScaleImage,bigScaleImage,parameters.overlapThresh);
    
    % correction with UNet
    correctedImage = correctWithUnet(mergedImage, probMap, parameters.probThresh, parameters.erosionRadius, parameters.dilationRadius, parameters.minOverlap, parameters.maxVParam, parameters.cAreaParam);
    
    imwrite(correctedImage, fullfile(outFolder.name, [imageID outFolder.ext]));
end
