function  addImagesToFolder( target, imageList, newFeatures)
%addImagesToFolder
%   adds to imageList's ID's to the target folder's 'database' of basic
%   features.
%   newFeatures is the matrix to be added
%   WARNING: the function simply adds the images, doesn't execute check if
%   the imageID already exist.

targetDB = fullfile(target,'basicFeatures.mat');
if exist(targetDB,'file')
    load(targetDB);
    [ features,imageNames ] = extendFeatures( features,imageNames,newFeatures,imageList); %#ok<ASGLU,NODEF> loaded and saved
    save(targetDB,'features','imageNames','featureNames');
end

targetCsv = fullfile(target,'features.csv');
if exist(targetCsv,'file')
    extendFeaturesCsv(targetCsv,newFeatures);
end


end

