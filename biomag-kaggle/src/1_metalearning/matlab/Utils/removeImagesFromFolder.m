function removedFeatures = removeImagesFromFolder( source, removeImageList)
%removedFeatures = removeImagesFromFolder( source, removeImageList)
%   Removes the images metadata from the source folder if it exists there.
%   Features is the array of the extracted features.
%   If basicFeatures.mat doesn't exist then features returned as empty
%   array.

sourceDB = fullfile(source,'basicFeatures.mat');
if exist(sourceDB,'file')
    load(sourceDB);
    
    removeIndex = zeros(1,length(imageNames)); %#ok<NODEF> loaded
    for i=1:length(imageNames)
        if any(1-cellfun(@isempty,strfind(removeImageList,imageNames{i})))
            removeIndex(i) = 1;
        end
    end
    removeIndex = logical(removeIndex);
    removedFeatures = features(removeIndex); %#ok<NODEF> loaded in
    features(removeIndex) = [];    
    imageNames(removeIndex) = [];     %#ok<NASGU> used in save
    
    save(sourceDB,'features','imageNames','featureNames');    
    
    sourceCsv = fullfile(source,'features.csv');
    if exist(sourceCsv,'file')        
        csvwrite(sourceCsv,features);
    end

end


end

