function [ result ] = cmppred( binMask, probMap, intRatioThresh )
%CMPPRED Summary of this function goes here
%   Input 
%   binMask: a binary mask image representing a single predicted object.
%   probMap: the probability map image
%   return: true if the nPixels(intersect(thresh(probmap, probMapThresh)))/nPixels(binMask) > intRatio 

    probMapMax=65535.0;

    % invert the probmaps
    probMap = probMapMax-probMap;
    
    weightedMask = double(binMask>0).*double(probMap./probMapMax);
    sumWeighted = sum(sum(weightedMask));
    sumBinMask = sum(sum(uint8(binMask>0)));
    result = (sumWeighted/sumBinMask)>intRatioThresh;
end

