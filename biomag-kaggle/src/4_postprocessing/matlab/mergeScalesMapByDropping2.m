function mergedImagesMap = mergeScalesMapByDropping2(smallScaleImages, bigScaleImages, overlapThresh)

mergedImagesMap = containers.Map;

imageIDs = smallScaleImages.map.keys();
smallScaleImagesMap = smallScaleImages.map;
bigScaleImagesMap = bigScaleImages.map;

for i=1:length(imageIDs)
    imageID = imageIDs{i};
    mergedImgSmall = smallScaleImagesMap(imageID);
    mergedImgBig = bigScaleImagesMap(imageID);
    
    mergedImg = mergeTwoMasksByDropping(mergedImgSmall, mergedImgBig, overlapThresh);
    
    mergedImagesMap(imageID) = relabelImage(mergedImg);
end
