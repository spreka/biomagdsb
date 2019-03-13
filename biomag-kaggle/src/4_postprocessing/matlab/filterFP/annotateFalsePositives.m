function annotation = annotateFalsePositives(gtDir, segmDir)

imagesList = dir(fullfile(gtDir,'*.png'));
if isempty(imagesList)
    imagesList = dir(fullfile(gtDir,'*.tiff'));
end

annotation = containers.Map();

for i=1:length(imagesList)
    imageName = imagesList(i).name;
    [~,imageBaseName,~] = fileparts(imageName);
    gtMask = imread(fullfile(gtDir,imageName));
    
    if exist( fullfile(segmDir,[imageBaseName '.tiff']), 'file' )
        predMask = imread(fullfile(segmDir,[imageBaseName '.tiff']));
        predMask = relabelImage(predMask);
        
        [~, ~, FPs] = evalSegmentedObjects(gtMask,predMask);
        
    end
    
    annotation(imageBaseName) = FPs;
end
