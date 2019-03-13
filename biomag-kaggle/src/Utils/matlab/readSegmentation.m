function segmentationMap = readSegmentation(segmFolder, imgExt)
%readSegmentation(segmFolder, imgExt) Reads each image in segmFolder as
%labelled segmentations and stores it in a containers.Map object. The keys
%are the image ids.

imagesList = dir(fullfile(segmFolder, ['*' imgExt]));

segmentationMap = containers.Map;

fprintf('Reading files from directory %s...\n',segmFolder);
numOfHashes = 15;
printedHashes = 0;
numOfImages = length(imagesList);

for i=1:numOfImages
    if i>=(printedHashes+1)*numOfImages/numOfHashes
        fprintf('#');
        printedHashes = printedHashes+1;
    end
    
    imgName = imagesList(i).name;
    [~,imageID,~] = fileparts(imgName);
    segmentationMap(imageID) = imread(fullfile(segmFolder, imgName));
end
fprintf('\n');
