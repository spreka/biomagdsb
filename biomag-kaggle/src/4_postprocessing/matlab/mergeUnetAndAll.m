function outImageMap = mergeUnetAndAll(smallScaleImagesMap, bigScaleImagesMap, sumProbMap, overlapThresh, probThresh, erosionRadius, dilationRadius, minSize, minOverlap)

allKeys = smallScaleImagesMap.keys();

% discard small objects 1st round
for i=1:length(allKeys)
    smallScaleImage = removeSmallObjects(smallScaleImagesMap(allKeys{i}), minSize);
    smallScaleImagesMap(allKeys{i}) = imresize(smallScaleImage, size(sumProbMap(allKeys{i})), 'nearest');
    bigScaleImage = removeSmallObjects(bigScaleImagesMap(allKeys{i}), minSize);
    bigScaleImagesMap(allKeys{i}) = imresize(bigScaleImage, size(sumProbMap(allKeys{i})), 'nearest');
end

%merge files from 2 scales
% mergedImgMap = mergeScalesMap2(struct('map',smallScaleImagesMap, 'scale', 1), struct('map',bigScaleImagesMap, 'scale', 2), scaleThrsh);
mergedImgMap = mergeScalesMapByDropping2(struct('map',smallScaleImagesMap, 'scale', 1), struct('map',bigScaleImagesMap, 'scale', 2),overlapThresh);

outImageMap = correctWithUnet2(mergedImgMap, sumProbMap, probThresh, erosionRadius, dilationRadius, minOverlap);




