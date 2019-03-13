function [ features,imageNames ] = extendFeatures( features,imageNames,newFeatures,newImageNames)
%extendFeatures Adds additional entries to the features array and the
%imageNames cellarray

features = [features; newFeatures];
imageNames = [imageNames newImageNames];

end

