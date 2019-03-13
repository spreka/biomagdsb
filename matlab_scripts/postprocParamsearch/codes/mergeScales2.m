function mergedImagesMap = mergeScales2(mrcnnScaleSmallFolder, mrcnnScaleBigFolder, scaleThresh, imgExt)

mergedImagesMap = containers.Map;

if nargin<4
    imgExt = '.tiff';
end

imageList = dir(fullfile(mrcnnScaleSmallFolder.name, ['*' imgExt]));

for i=1:length(imageList)
    [~,imageID,~] = fileparts(imageList(i).name);
    mergedImg = imread(fullfile(mrcnnScaleSmallFolder.name, imageList(i).name));
    [medianSize, stdSize] = estimateCellSize2(mergedImg);
    medianSize = medianSize / mrcnnScaleSmallFolder.scale;
%     stdSize = stdSize / mrcnnScaleSmallFolder.scale;
    
    if medianSize < scaleThresh
    	mergedImg = imread(fullfile(mrcnnScaleBigFolder.name, imageList(i).name));
    end
    
    mergedImagesMap(imageID) = relabelImage(mergedImg);
end