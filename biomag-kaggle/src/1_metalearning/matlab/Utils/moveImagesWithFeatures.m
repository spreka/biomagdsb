function moveImagesWithFeatures( source,target,fileList )
%moveImagesWithFeatures
%   Moves images with it's estimated features from one folder to another
%   one.
%   
%   If in the destination folder there is already a basicFeatures.mat then
%   the copied images are also added to that, as well as to the
%   features.csv
%   If in the source folder there is a basicFeatures.mat, then the moved
%   images are removed from that.
%   The above applies also to the features.csv.
%
%   INPUTS:
%       source      Source folder
%       target      Target folder
%       fileList    fileList the image names (with extention) to move from
%                   source to target

%{
features = removeImagesFromFolder(source,fileList);

if isempty(features)
    %TODO: make folder to calculate basic features and then remove
end

addImagesToFolder( target, fileList, features);

%}
%and move actually the images
for i=1:numel(fileList)
    movefile(...
        fullfile(source,fileList{i}),...
        fullfile(target,fileList{i})...
        )
end


end

