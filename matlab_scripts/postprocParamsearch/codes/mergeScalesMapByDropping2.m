function mergedImagesMap = mergeScalesMapByDropping2(smallScaleImages, bigScaleImages, overlapThresh, areaThresh, medianSizeThresh)

mergedImagesMap = containers.Map;

imageIDs = smallScaleImages.map.keys();
smallScaleImagesMap = smallScaleImages.map;
bigScaleImagesMap = bigScaleImages.map;

for i=1:length(imageIDs)
        
    imageID = imageIDs{i};
    
    image1 = smallScaleImagesMap(imageID);
    image2 = bigScaleImagesMap(imageID);
                    
    [median_size, std_size] = estimateCellSize2(image1);
    if isnan(median_size)
        [median_size, std_size] = estimateCellSize2(image2);
    end
    
    if median_size > medianSizeThresh        
        mergedImagesMap(imageID) = image1;
    else            
        mergedImg = mergeTwoMasksByDropping(image1, image2, overlapThresh, areaThresh);   
        mergedImagesMap(imageID) = mergedImg;
    end
end
