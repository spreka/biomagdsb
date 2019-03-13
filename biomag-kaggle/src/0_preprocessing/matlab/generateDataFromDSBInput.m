function generateDataFromDSBInput(inputDir, varargin)
%GENERATEDATAFROMDSBINPUT Applies several preprocessing steps on original DSB data
%
% generateDataFromDSBInput(inputDir) collects the images from original DSB
% file structure, which is
%   inputDir/
%       imageId1/
%           images/
%               imageId1.png
%           masks/
%               maskId1.png
%               ...
%               maskIdn.png
%       ...
%
% inputDir contains the input images of train data.
%
% generateDataFromDSBInput(inputDir, Property, Value) makes possible to
% generate other types of output data. Property, Value pair settings are
% for costumization of output:
%
% > 'outputDir' - the folder where to put generated data. Default value out
% output folder is the same as input folder.
% > 'outputDataType' is a string having one of these values
%
%       'simpleCollect'               - copies all images into 1 folder
%
%       'mergedMaskImage'           - merges all nuclei masks into 1 binary
%                                     image
%
%       'mergedMaskLabelledImage'     - merges all nuclei masks into 1
%                                     labelled image, where labels have
%                                     value from 0 to number of nuclei
%
%       'outlinedImage'               - burn all nuclei boundaries to the
%                                     original image
%
%       'mergedBorderMask'            - merges nuclei border points into 1
%                                     binary image
%
%       'mergedInnerPointsMask'       - merges nuclei inner points into 1
%                                     binary image
%
%       'boundingBox'                 - saves bounding boxes of images to txt
%
%       'ilastikTrainMap'             - generates external training label
%                                     file based on train folder
%                                     options:
%                                               numberOfLabels - 2 or 3;
%                                               samplingRates - vector of
%                                               values between 0 and 1 of
%                                               size numberOfLabels;
%                                               dilationRadius - half with
%                                               of border
%
%       'buildDatabase'               - creates a csv file to save input file
%                                     structure
% > 'options' - a cell array with Name, Value pairs for output specific settings:
%
%   > 'ilastikTrainMap' output options: 'numberOfLabels' - 2 or 3;
%   'samplingRates' - sampling rates for the labels (2 or 3 length vector
%   of values [0..1]; 'dilationRadius' - width of band considered as
%   boundary area near to object boundaries.
%   > 'overlayContour' output options: 'colour' - RGB colour of overlayed
%   contours or one of the built in colours ('red', 'green', 'blue' and
%   'yellow').
%
%
% Example 1: collects all input images into a folder
%       generateDataFromDSBInput('/home/biomag/kaggle_data_science_bowl/data/stage1_train/',...
%                                'outputDir','/home/biomag/kaggle_data_science_bowl/data/',...
%                                'outputDataType','simpleCollect');
%
% Example 2: generates external training map for ilastik
%       generateDataFromDSBInput('/home/biomag/kaggle_data_science_bowl/data/stage1_train/',...
%                                'outputDir','/home/biomag/kaggle_data_science_bowl/data/',...
%                                'outputDataType','ilastikTrainMap',...
%                                'options',{'numberOfLabels',2,'samplingRates',[0.05 0.25 0.55],'dilationRadius',2});
%
%   See also createTrainingMapForIlastikFromMasks

defaultOutputDir = inputDir;
defaultOutputDataType = 'outlinedImage';
expectedOutputDataTypes = ...
    {'simpleCollect','outlinedImage', 'mergedMaskImage', 'mergedMaskLabelledImage',...
    'mergedBorderMask','mergedInnerPointsMask','boundingBox','ilastikTrainMap',...
    'buildDatabase', 'tivadarTrainMap'};

p = inputParser;
addRequired(p, 'inputDir', @(x) isdir(x));
addParameter(p, 'outputDir', defaultOutputDir, @ischar);
addParameter(p, 'outputDataType', defaultOutputDataType,...
    @(x) any(validatestring(x,expectedOutputDataTypes)));

addParameter(p, 'options', {}, @iscell);

parse(p, inputDir, varargin{:});

inputDirName = getDirNameFromPath(inputDir);

% list and filter input directory (process only directories)
imageList = dir(inputDir);
imageList(~cat(1, imageList.isdir)) = [];
for i=length(imageList):-1:1
    if strcmp(imageList(i).name,'.') || strcmp(imageList(i).name,'..')
        imageList(i) = [];
    end
end

% create output folder structure
outputDataDir = fullfile(p.Results.outputDir,sprintf('out_%s_%s',inputDirName, p.Results.outputDataType));
if ~exist(outputDataDir,'dir')
    mkdir(outputDataDir);
end


imageNum = length(imageList);
successfulProcessed = 0;

maskFileCount = 0;


%%%%%%%%%%%%%%%%%%%%%% main loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% iteration through images

for i=1:imageNum
    imageID = imageList(i).name;
    
    fprintf('[%d/%d] Generating ''%s'' output for image %s... \n',i,imageNum,p.Results.outputDataType,imageID);
    
    % TODO include database generation for all output types
    switch p.Results.outputDataType
        
        case 'simpleCollect'
            
            inputImage = imread(fullfile(inputDir,imageID,'images',sprintf('%s.png',imageID)));
            imwrite(inputImage, fullfile(outputDataDir,sprintf('%s.png',imageID)),'png',...
                'Alpha', ones(size(inputImage,1),size(inputImage,2)));
            
        case 'mergedMaskImage'
            
            [masks, ~] = readMasks(inputDir, imageID);
            if ~isempty(masks)
                mergedMasks = uint8(sum(cat(3,masks{:}),3)>0)*255;
                imwrite(mergedMasks, fullfile(outputDataDir, sprintf('%s.png',imageID)));
            else
                warning('generateData:noMasksExist','No masks are available for image %s', imageID);
            end
            
        case 'mergedMaskLabelledImage'
            
            [masks, ~] = readMasks(inputDir, imageID);
            if ~isempty(masks)
%                 mergedMasks = zeros(size(masks{1}));
                for mi=1:length(masks)
                    masks{mi} = uint16(masks{mi}>0)*mi;
%                     mergedMasks = mergedMasks + (masks{mi}>0) * mi;
                end
                %mergedMasks = sum(cat(3,masks{:}),3,'native');
                mergedMasks = max(cat(3,masks{:}),[],3);
                
                imwrite(uint16(mergedMasks), fullfile(outputDataDir, sprintf('%s.png',imageID)),'BitDepth',16);
            else
                warning('generateData:noMasksExist','No masks are available for image %s', imageID);
            end
            
        case 'mergedBorderMask'
            
            [masks, ~] = readMasks(inputDir, imageID);
            if ~isempty(masks)
                masks = cellfun(@(x) bwperim(x>0), masks, 'UniformOutput',false);
                mergedBorders = uint8(any(cat(3,masks{:}),3)>0)*255;
                imwrite(mergedBorders, fullfile(outputDataDir, sprintf('%s.png',imageID)));
            else
                warning('generateData:noMasksExist','No masks are available for image %s', imageID);
            end
            
        case 'mergedInnerPointsMask'
            
            [masks, ~] = readMasks(inputDir, imageID);
            if ~isempty(masks)
                borders = cellfun(@(x) uint8(bwperim(x>0))*255, masks, 'UniformOutput',false);
                innerPointsMask = cellfun(@(m,c) m-c, masks, borders,'UniformOutput',false);
                mergedInnerPoints = uint8(any(cat(3,innerPointsMask{:}),3)>0)*255;
                imwrite(mergedInnerPoints, fullfile(outputDataDir, sprintf('%s.png',imageID)));
            else
                warning('generateData:noMasksExist','No masks are available for image %s', imageID);
            end
            
        case 'outlinedImage'
            
            inputImage = readInputImage(inputDir, imageID);
            [masks, ~] = readMasks(inputDir, imageID);
            if ~isempty(masks)
                masks = cellfun(@(x) bwperim(x>0), masks, 'UniformOutput',false);
                mergedBorders = uint8(any(cat(3,masks{:}),3)>0);
            else
                mergedBorders = uint8(zeros(size(inputImage,1), size(inputImage,2)));
            end
            % TODO change contour from options
            outputImage = overlayContour(inputImage, mergedBorders, p.Results.options);
            imwrite(outputImage, fullfile(outputDataDir, sprintf('%s.png',imageID)));
            
        case 'boundingBox'
            
            [masks, ~] = readMasks(inputDir, imageID);
            boundingBoxes = cellfun(@getBoundingBoxes, masks, 'UniformOutPut',false);
            boundingBoxes = cell2mat(boundingBoxes);
            dlmwrite(fullfile(outputDataDir, sprintf('%s.txt',imageID)), boundingBoxes,...
                'delimiter', ' ');
            
        case 'ilastikTrainMap'
            
            [masks, ~] = readMasks(inputDir, imageID);
            if ~isempty(masks)
                trainingMap = createTrainingMapForIlastikFromMasks(masks, 'options', p.Results.options);
                imwrite(trainingMap, fullfile(outputDataDir, sprintf('%s.png',imageID)));
            end
            
        case 'tivadarTrainMap'
            
            inputImage = readInputImage(inputDir, imageID);
            mkdir(fullfile(outputDataDir, imageID, 'images'));
            imwrite(inputImage, fullfile(outputDataDir, imageID, 'images', sprintf('%s.png',imageID)));
            
            [masks, ~] = readMasks(inputDir, imageID);
            if ~isempty(masks)
                trainingMap = createTrainingMapForIlastikFromMasks(masks, 'options', p.Results.options);
                mkdir(fullfile(outputDataDir, imageID, 'masks'));
                imwrite(trainingMap, fullfile(outputDataDir, imageID, 'masks', sprintf('%s.png',imageID)));
            end
            
        case 'buildDatabase'
            
            if i==1
                maskIdCounter = 0;
                databaseFileId = fopen(fullfile(outputDataDir,'database.csv'),'w');
                fprintf(databaseFileId,',name,image_path,mask_path\n');
            else
                databaseFileId = fopen(fullfile(outputDataDir,'database.csv'),'a');
            end
            maskFileNames = getMaskFileNames(inputDir, imageID);
            if ~isempty(maskFileNames)
                for mi=1:length(maskFileNames)
                    fprintf(databaseFileId, '%d,%s,%s,%s\n',...
                        maskIdCounter+mi,...
                        imageID,...
                        fullfile(inputDir,'images',sprintf('%s.png',imageID)),...
                        fullfile(inputDir,'masks',maskFileNames{mi}));
                end
                maskIdCounter = maskIdCounter + length(maskFileNames);
            else
                fprintf(databaseFileId, '%d,%s,%s,\n',...
                    maskIdCounter,...
                    imageID,...
                    fullfile(inputDir,'images',sprintf('%s.png',imageID)));
                maskIdCounter = maskIdCounter +1;
            end
            fclose(databaseFileId);
        case 'patchImages'
            error('generateData:createPatchImages', 'Generating patch images functionality is not implemented yet.');
%             options = p.Results.options;
%             if ~isfield(options, 'patchSize')...
%                     && ~validateattributes(options.patchSize,{'numeric','integer'},{'numel',2})
%                 options.patchSize = [50 50];
%             end
%             if ~isfield(options, 'overlapLevel')...
%                     && ~validateattributes(options.patchSize,{'numeric'},{'numel',2})
%                 options.overlapLevel = [0.1 0.1];
%             end
    end
    
    successfulProcessed = successfulProcessed + 1;
end

%%%%%%%%%%%%%%%%%%%%% end main loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('%d images were processed.\n',successfulProcessed);

function inputImage = readInputImage(inputDir, imageID)
try
    inputImage = imread(fullfile(inputDir,imageID,'images',sprintf('%s.png',imageID)));
catch e
    error('generateData:readInputImage', 'Error while reading image %d.png\n',imageID);
end

function [masks, maskFileNames] = readMasks(inputDir, imageID)
maskList = dir(fullfile(inputDir, imageID, 'masks', '*.png'));
maskFileNames = arrayfun(@(x) x.name, maskList, 'UniformOutput', false);
masks = cell(length(maskList),1);
for i=1:length(maskList)
    try
        masks{i} = imfill(imread(fullfile(inputDir,imageID,'masks',maskList(i).name)),'holes');
    catch e
        error('generateData:readMasks', 'Error while reading mask %s for image %s.png\n',maskList(i).name, imageID);
    end
end

function maskFileNames = getMaskFileNames(inputDir, imageID)
maskList = dir(fullfile(inputDir, imageID, 'masks', '*.png'));

maskFileNames = arrayfun(@(x) x.name, maskList, 'UniformOutput', false);

function bbxs = getBoundingBoxes(mask)
% [h,w] = size(mask);
props = regionprops(logical(mask),'BoundingBox');
if ~isempty(props)
    bbxs = ceil(cat(1,props.BoundingBox));
    bbxs(:,4) = bbxs(:,2)+bbxs(:,4)-1;
    bbxs(:,3) = bbxs(:,1)+bbxs(:,3)-1;
    bbxs = bbxs(:,[2 1 4 3]);
else
    bbxs = [];
end

function outlinedImage = overlayContour(inputImage, contours, options)
if isempty(options)
    colour = [1 0 0];
else
    for i=1:2:length(options)
        if strcmp(options{i},'colour')
            colour = options{i+1};
        end
    end
end
if ischar(colour)
    switch colour
        case 'red'
            c = [1 0 0];
        case 'green'
            c = [0 1 0];
        case 'blue'
            c = [0 0 1];
        case 'yellow'
            c = [1 1 0];
        otherwise
            error('generateData:overlayContour:noSuchColourImplemented','There is no built in colour %s.',colour );
    end
else
    if length(colour)~=3
        error('Colour value must be a 3 element numeric vector or a built in colour (red, green, blue or yellow).');
    else
        c = colour;
    end
end

[~,~,d] = size(inputImage);
if d==1
    outlinedImage = repmat(inputImage .* uint8(~contours),1,1,3);
else
    outlinedImage = inputImage .* repmat(uint8(~contours),1,1,3);
end
outlinedImage = outlinedImage + cat(3,contours*c(1)*255,contours*c(2)*255,contours*c(3)*255);

function dirName = getDirNameFromPath(fullPath)
if strcmp(fullPath(end), filesep)
    fullPath = fullPath(1:end-1);
end
if ~isdir(fullPath)
    error('generateData:getDirName:notDirectory', '%s is not a directory.', fullpath);
else
    [~, b, c] = fileparts(fullPath);
    dirName = [b,c];
end
