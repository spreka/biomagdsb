function writeSegmentation(segmentationMap, segmFolder, imgExt)
%readSegmentation(segmentationMap, segmFolder, imgExt) Writes each image in 
% segmentationMap to segmFolder.

fprintf('Writing files to directory %s...\n',segmFolder);
numOfHashes = 15;
printedHashes = 0;

imageIDs = segmentationMap.keys;
numOfImages = numel(imageIDs);

if ~exist(segmFolder, 'dir')
    mkdir(segmFolder);
end

% NOTE: parfor does not improve here
for i=1:numOfImages
    if i>=(printedHashes+1)*numOfImages/numOfHashes
        fprintf('#');
        printedHashes = printedHashes+1;
    end
    imageName = [imageIDs{i} imgExt];
    imwrite(uint16(segmentationMap(imageIDs{i})), fullfile(segmFolder, imageName));
end

fprintf('\n');
