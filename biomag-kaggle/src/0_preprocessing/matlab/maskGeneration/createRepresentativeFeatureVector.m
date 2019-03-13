function [ reprFeature ] = createRepresentativeFeatureVector( segmentedImage, featureDataBase)
 %AUTHOR:   Abel Szkalisity
% DATE:     10 March, 2018
% NAME:     createRepresentativeFeatureVector
%
%   Gives back a representative feature vector for a given segmentedImage
%
%   INPUT:
%       segmentedImage: a grayscale image that stores object labels, the
%       basis of the search in the database
%       featureDataBase: a database that stores features of many really
%       annotated nuclei. It must be accessed only via its functions.
%       
%   OUTPUT:
%       a 2xm array where currently m=5 and includes features in the
%       following order: Area, ShapeRatio (Major/Minor axis length), and
%       circularity, eccentricity, solidity The first row is the mean value the second is the
%       standard deviation.
%

N = 100;

    vectors = FeatureDataBase.measureCellProperties(segmentedImage,{'Area','ShapeRatio','Circularity','Eccentricity','Solidity'});

    targetVectors = vectors(:,3:5);

    [~,~,~,hitVectors] = featureDataBase.fetchCloseIndex({'Circularity','Eccentricity','Solidity'},targetVectors,N);

reprFeature = zeros(2,size(vectors,2));
reprFeature(1,1:2) = mean(vectors(:,[1,2]));
reprFeature(2,1:2) = std(vectors(:,[1,2])); % the first 2 is measured on the image i.e. we trust in the initial segmentation in that sense
reprFeature(1,3:5) = mean(hitVectors,1);
reprFeature(2,3:5) = std(hitVectors,1); % average other values are extracted from the cells


end

