function mergedImg = mergeScales(scalesFolders, imageName)

scales = [scalesFolders.scale];

[sortedScales, ind] = sort(scales);

sortedScalesFolders = scalesFolders(ind);

[~,imageID,~] = fileparts(imageName);
% scalefolders(1) should belong to the original sized images
[medianSize, stdSize] = estimateCellSize(sortedScalesFolders(1).name, imageID, 1);
medianSize = medianSize / sortedScalesFolders(1).scale;
stdSize = stdSize / sortedScalesFolders(1).scale;

if medianSize < 5 % biggest scale % TODO change threshold
    scale = sortedScales(end);
elseif medianSize < 45 % mid scale % TODO change threshold
    scale = sortedScales(end-1);
else
    scale = sortedScales(1); % original
end

inImg = imread(fullfile(sortedScalesFolders( sortedScales==scale ).name, imageName));
% mergedImg = imresize(inImg, 1./scale, 'nearest');
mergedImg = inImg;

mergedImg = relabelImage(mergedImg);
