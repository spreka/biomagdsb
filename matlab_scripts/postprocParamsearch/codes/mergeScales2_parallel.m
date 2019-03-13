function mergedImagesMap = mergeScales2_parallel(mrcnnScaleSmallFolder, mrcnnScaleBigFolder, scaleThresh, imgExt)

if nargin<4
    imgExt = '.tiff';
end

imageList = dir(fullfile(mrcnnScaleSmallFolder.name, ['*' imgExt]));
imageListCellArray = {imageList.name};
imagesCellArray = cell(length(imageListCellArray),1);

% NOTE: parfor does not improve here too much for 65 images
parfor i=1:length(imageList)
%     [~,imageID,~] = fileparts(imageList(i).name);
    mergedImg = imread(fullfile(mrcnnScaleSmallFolder.name, imageList(i).name));
    [medianSize, stdSize] = estimateCellSize2(mergedImg);
    medianSize = medianSize / mrcnnScaleSmallFolder.scale;
%     stdSize = stdSize / mrcnnScaleSmallFolder.scale;
    
    if medianSize < scaleThresh
    	mergedImg = imread(fullfile(mrcnnScaleBigFolder.name, imageList(i).name));
    end
    imagesCellArray{i} = mergedImg;
end

mergedImagesMap = containers.Map;
for i=1:length(imageList)
    [~,imageID,~] = fileparts(imageList(i).name);
    mergedImagesMap(imageID) = relabelImage(imagesCellArray{i});
end
