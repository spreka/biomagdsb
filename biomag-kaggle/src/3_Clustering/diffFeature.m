function [ F ] = diffFeature( feature1,feature2 )
%generates a feature vector that describes the difference between the 2
%input feature vector, that has to be the same long
%   Used for keeping the same difference measure in training and predicting
%   image distances for clustering. Kaggle DSB 2018

F = abs(feature1-feature2);


end

