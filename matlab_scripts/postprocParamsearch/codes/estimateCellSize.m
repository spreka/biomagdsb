function [median_size, std_size] = estimateCellSize(folderName, imageID, varargin)

if nargin<3
    labelledInput = 1;
else
    labelledInput = varargin{1};
end

if ~labelledInput
    % mask images
    inImageList = dir(fullfile(folderName, 'masks','*.png'));

    allSize = zeros(numel(inImageList), 1);
    for k=1:numel(inImageList)
        inImage = imread(fullfile(folderName, 'masks', inImageList(k).name)); 
        props = regionprops(inImage>0, 'EquivDiameter'); 
        allSize(k) = max([props.EquivDiameter]);    
    end
else
    inImageList = dir(fullfile(folderName, [imageID, '*.tiff']));
    if ~isempty(inImageList)
        inImage = imread(fullfile(folderName, inImageList(1).name));
        props = regionprops(inImage, 'EquivDiameter');
        props( isempty(cat(1,props.EquivDiameter)) ) = [];
        allSize = cat(1,props.EquivDiameter);     
    end
end

median_size = median(allSize);
std_size = std(allSize);
