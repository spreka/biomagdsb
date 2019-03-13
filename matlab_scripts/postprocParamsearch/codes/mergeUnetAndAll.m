function outImageMap = mergeUnetAndAll(smallScaleImagesMap, bigScaleImagesMap, sumProbMap, overlapThresh, probThresh, erosionRadius, dilationRadius, minSize, minOverlap, maxVParam, cAreaParam, areaThresh, medianSize)

allKeys = smallScaleImagesMap.keys();

% discard small objects 1st round
for i=1:length(allKeys)
                
    %fprintf('Key: %s\n', allKeys {i});
    %fprintf('isKey: %i\n', smallScaleImagesMap.isKey(allKeys{i}));
    %fprintf('isKey: %i\n', bigScaleImagesMap.isKey(allKeys{i}));
        
    %tfS = smallScaleImagesMap.isKey(allKeys{i});    
    %tfM = bigScaleImagesMap.isKey(allKeys{i});    
    %if (tfS == false || tfM == false)
    %    continue;        
    %end
            
    %[x1,y1,z1] = size(smallScaleImage);
    %fprintf('Size: %i %i %i\n', x1, y1, z1);

    smallScaleImage = removeSmallObjects(smallScaleImagesMap(allKeys{i}), minSize);
    smallScaleImagesMap(allKeys{i}) = imresize(smallScaleImage, size(sumProbMap(allKeys{i})), 'nearest');
    bigScaleImage = removeSmallObjects(bigScaleImagesMap(allKeys{i}), minSize);
    bigScaleImagesMap(allKeys{i}) = imresize(bigScaleImage, size(sumProbMap(allKeys{i})), 'nearest');
end

%writeSegmentation(smallScaleImagesMap, 'd:\kaggle\temp\1x', '.tiff');
%writeSegmentation(bigScaleImagesMap, 'd:\kaggle\temp\2x', '.tiff');

%merge files from 2 scales
% mergedImgMap = mergeScalesMap2(struct('map',smallScaleImagesMap, 'scale', 1), struct('map',bigScaleImagesMap, 'scale', 2), scaleThrsh);
mergedImgMap = mergeScalesMapByDropping2(struct('map', smallScaleImagesMap, 'scale', 1), struct('map', bigScaleImagesMap, 'scale', 2), overlapThresh, areaThresh, medianSize);
%writeSegmentation(mergedImgMap, '/media/baran/LinuxData/Downloads/Challange/Optimizer/temp', '.tiff');

%outImageMap = mergedImgMap;
%return;

outImageMap = correctWithUnet2(mergedImgMap, sumProbMap, probThresh, erosionRadius, dilationRadius, minOverlap, maxVParam, cAreaParam);
%writeSegmentation(outImageMap, '/media/baran/LinuxData/Downloads/Challange/Optimizer/temp2', '.tiff');




