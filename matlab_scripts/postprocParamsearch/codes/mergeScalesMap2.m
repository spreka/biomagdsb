function mergedImagesMap = mergeScalesMap2(smallScaleImages, bigScaleImages, scaleThresh)
%choses a scale and erodes with 1 pixel

mergedImagesMap = containers.Map;

if nargin<4
    imgExt = '.tiff';
end

imageIDs = smallScaleImages.map.keys();
smallScaleImagesMap = smallScaleImages.map;
bigScaleImagesMap = bigScaleImages.map;

for i=1:length(imageIDs)
    imageID = imageIDs{i};
    mergedImg = smallScaleImagesMap(imageID);
    [medianSize, stdSize] = estimateCellSize2(mergedImg);
    disp([imageID ': ' num2str(medianSize)]);
    medianSize = medianSize / smallScaleImages.scale;
    stdSize = stdSize / smallScaleImages.scale;
    
    if medianSize > scaleThresh
    	mergedImg = bigScaleImagesMap(imageID);
    end
    mergedImagesMap(imageID) = mergedImg;
end
