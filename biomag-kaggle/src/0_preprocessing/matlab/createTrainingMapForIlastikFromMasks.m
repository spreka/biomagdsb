function trainLabelMap = createTrainingMapForIlastikFromMasks(masks,varargin)
%createTrainingMapForIlastik generates an external file for ilastik to
% import training labels.
%
% trainLabelMap = createTrainingMapForIlastikFromMasks(masks) generates a
% train label map for ilastik with 2 labels (for BG and FG respectively)
% with sampling rates 0.05 for BG, 0.25 for FG inner points and 0.55 for FG
% contour points. masks is a cell array of HxW binary images containing 1
% object per image. trainLabelMap is a HxW labeled matrix containing 2
% unique labels.
% 
% trainLabelMap = createTrainingMapForIlastikFromMasks(masks,'options',optionsCellArray)
% generates costumized train label map. Available Name, Value pairs:
%
% > 'numberOfLabels' - 2, 3 or 4.
% > 'samplingRates' - 2 or 3 length vector of values between [0, 1]. 1st
% element is for BG sampling rate, 2nd element is for FG, 3rd element is
% for BORDER area (or equals to FG sampling rate if does not exist).
% > 'dilationRadius' - width of band considered as 
% boundary area near to object boundaries.

defaultNumberOfLabels = (2);
defaultSamplingRate = [0.05 0.25 0.55]; % BG-INNER-BORDER

p = inputParser;
p.KeepUnmatched = true;
addRequired(p, 'masks', @iscell);

% addParameter(p, 'numberOfLabels', defaultNumberOfLabels, @(x) validateattributes(x,{'numeric'},{'>=',2,'<=',3}));
% addParameter(p, 'samplingRates', defaultSamplingRate, @isnumeric);

validateoptionsfun = @(x) iscell(x) && mod(numel(x),2)==0;
addParameter(p, 'options', {}, validateoptionsfun);
parse(p, masks, varargin{:});
options = struct('numberOfLabels', defaultNumberOfLabels,...
                 'samplingRates', defaultSamplingRate,...
                 'dilationRadius', 1,...
                 'labelSize',1);
             
for i=1:length(p.Results.options)/2
    options.(p.Results.options{i*2-1}) = p.Results.options{i*2};
end

samplingRates = options.samplingRates;
% if numel(samplingRates)<3
    samplingRates = [samplingRates repmat(samplingRates(end),1,options.numberOfLabels-numel(samplingRates))];
% end

if isfield(options,'dilationRadius')
    SE = strel('disk',options.dilationRadius,4);
else
    SE = strel('disk',1,4);
end

mergedMasks = uint8(sum(cat(3,masks{:}),3)>0); 
trainLabelMap = uint8(zeros(size(mergedMasks)));
propsMM = regionprops(mergedMasks>0,'PixelIdxList');
mergedMaskPointsIdxList = cat(1,propsMM.PixelIdxList);

props = regionprops(mergedMasks==0,'PixelIdxList'); 
bgPointsIdxList = cat(1,props.PixelIdxList);

% prepare merged border points
borders = cellfun(@(x) bwperim(x>0), masks, 'UniformOutput',false);
mergedBorders = any(cat(3,borders{:}),3)>0;

dilatedBorders = imdilate(mergedBorders, SE);
propsDB = regionprops(dilatedBorders>0,'PixelIdxList');
dilatedBorderPointsIdxList = cat(1,propsDB.PixelIdxList);
numberOfDilatedBorderPoints = numel(dilatedBorderPointsIdxList);

% remove points near to object boundaries from background points
bgPointsOnlyIdxList = setdiff( bgPointsIdxList, dilatedBorderPointsIdxList);
numberOfBGPointsOnly = numel(bgPointsOnlyIdxList);

% remove points near to object boundaries from object points
innerPointsIdxList = setdiff(mergedMaskPointsIdxList, dilatedBorderPointsIdxList);
numberOfInnerPoints = numel(innerPointsIdxList);

if options.numberOfLabels==2
    if samplingRates(1)<1.0
        trainLabelMap(bgPointsOnlyIdxList(randperm(numberOfBGPointsOnly,int16(numberOfBGPointsOnly*samplingRates(1))))) = 1;
    else
        trainLabelMap(bgPointsIdxList) = 1;
    end
    bgBorderPoints = intersect(bgPointsIdxList, dilatedBorderPointsIdxList);
    numberOfBGBorderPoints = numel(bgBorderPoints);
    if samplingRates(3)<1.0
        trainLabelMap(bgBorderPoints(randperm(numberOfBGBorderPoints,int16(numberOfBGBorderPoints*samplingRates(3))))) = 1;
    else
        trainLabelMap(bgBorderPoints) = 1;
    end
    
    if samplingRates(2)<1.0
        trainLabelMap(innerPointsIdxList(randperm(numberOfInnerPoints,int16(numberOfInnerPoints*samplingRates(2))))) = 2; 
    else
        trainLabelMap(innerPointsIdxList) = 2;
    end

    innerBorderPointsIdxList = intersect(mergedMaskPointsIdxList, dilatedBorderPointsIdxList);
    numberOfInnerBorderPoints = numel(innerBorderPointsIdxList);
    if samplingRates(3)<1.0
        trainLabelMap(innerBorderPointsIdxList(randperm(numberOfInnerBorderPoints,int16(numberOfInnerBorderPoints*samplingRates(3))))) = 2;
    else
        trainLabelMap(innerBorderPointsIdxList) = 2;
    end
    
elseif options.numberOfLabels==3
% 3 classes: BG(1), FG(2), BORDER(3)
% dilated border area is probably not a good option for sharp edges
    if samplingRates(1)<1.0
        trainLabelMap(bgPointsOnlyIdxList(randperm(numberOfBGPointsOnly,int16(numberOfBGPointsOnly*samplingRates(1))))) = 1;
    else
        trainLabelMap(bgPointsOnlyIdxList) = 1;
    end
    
    if samplingRates(2)<1.0
        trainLabelMap(innerPointsIdxList(randperm(numberOfInnerPoints,int16(numberOfInnerPoints*samplingRates(2))))) = 2; 
    else
        trainLabelMap(innerPointsIdxList) = 2;
    end
    
    if samplingRates(3)<1.0
        trainLabelMap(dilatedBorderPointsIdxList(randperm(numberOfDilatedBorderPoints,int16(numberOfDilatedBorderPoints*samplingRates(3))))) = 3;
    else
        trainLabelMap(dilatedBorderPointsIdxList) = 3;
    end
    
elseif options.numberOfLabels==4 
% case of 4 classes: BG, FG, SIMPLE-BORDER, INNER-BORDER (splits objects)    
    if samplingRates(1)<1.0
        trainLabelMap(bgPointsIdxList(randperm(numberOfBGPointsOnly,int16(numberOfBGPointsOnly*samplingRates(1))))) = 1;
    else
        trainLabelMap(bgPointsIdxList) = 1;
    end
    
    if samplingRates(2)<1.0
        trainLabelMap(innerPointsIdxList(randperm(numberOfInnerPoints,int16(numberOfInnerPoints*samplingRates(2))))) = 2; 
    else
        trainLabelMap(innerPointsIdxList) = 2;
    end
    
    if samplingRates(3)<1.0
        trainLabelMap(dilatedBorderPointsIdxList(randperm(numberOfDilatedBorderPoints,int16(numberOfDilatedBorderPoints*samplingRates(3))))) = 3;
    else
        trainLabelMap(dilatedBorderPointsIdxList) = 3;
    end
    
    dilatedBackground = imdilate(mergedMasks==0, strel('disk',1,4));
    propsDilatedBackground = regionprops(dilatedBackground, 'PixelIdxList');
    innerBorderPointsIdxList = setdiff(dilatedBorderPointsIdxList, cat(1,propsDilatedBackground.PixelIdxList));
    numberOfInnerBorderPoints = numel(innerBorderPointsIdxList);
    
    if samplingRates(4)<1.0
        trainLabelMap(innerBorderPointsIdxList(randperm(numberOfInnerBorderPoints,int16(numberOfInnerBorderPoints*samplingRates(4))))) = 4;
    else
        trainLabelMap(innerBorderPointsIdxList) = 4;
    end
else
    %     error('createTrainingMap:notImplementedLabelNumber','Training map generation for %d labels is not implemented yet.', options.numberOfLabels);
end