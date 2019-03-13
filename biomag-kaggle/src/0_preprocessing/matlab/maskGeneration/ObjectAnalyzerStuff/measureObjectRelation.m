function [ relationFeatures, featureNames] = measureObjectRelation(m1,m2,k)
%function [ relationFeatures, featureNames] = measureObjectRelation(mask1, mask2, mask1Centers, mask2Centers, object1MaxRadius, object2MaxRadius, mask1PixelChains, mask2PixelChains, k)
% AUTHOR:   Abel Szkalisity
% DATE:     Nov 28, 2016
% NAME:     measureObjectRelation
%
% A general function to measure relations between two type of object. By
% relation we mean their relative localization. This function computes some
% relation measurements for each object in mask1.
%
% INPUTS:
%  m1,m2            The first 2 input is the same in type, they describe
%                   the objects of interes. This is a structure with
%                   several fields (not only the actual mask image is
%                   passed as input but several other properties to fasten
%                   computation.)
%                   Fields for m1 and m2:
%                       Mask: Image that identifies the objects to measure.
%                       The pixels with the same mask image value form an
%                       object. The mask image can be provided in a layered
%                       form (3D mask image). Still in this case the indices
%                       identifying the objects are unique within the complete
%                       image. The indices identifies the objects, going from 1
%                       to N1 consecutively. m1.Mask and m2.Mask must have
%                       the same dimension.
%                       Centers: an N1 by 2 matrix for the centroid
%                       positions of the objects.
%                       MaxRadius: an N1 long array, providing the maximal
%                       distance between any contour point and the centre
%                       of the object.
%                       PixelChains: an N1 long cellarray. For each object
%                       it contains a matrix where each row is one pixel of
%                       the contour chain.
%  k                Parameter for k nearest neighbours.
%  
% OUTPUT:
%  relationFeatures An N1 by m matrix where m is the number of relation
%                   describing features defined by this function.
%                   Currently calculated features:
%                       1. Overlap ratio: commonArea/object1Area
%                       2. mixture index: |kNN = 2|/k (2 stands for object type 2 to which we compare)
%                       3. Closest distance: How far is the closest other
%                       typed object to this one (distances measured from center)
%                       4. Closest contour distance: how far is the closest
%                       object if the distances are measured from the
%                       contours?
%                       5. Mean contour distance: for each contour point of
%                       the object1 calculate the closest contour point of
%                       any type2 object and take the mean of these.
%                   The rows of this matrix corresponds to the index of
%                   mask1.
%                   Description terminology: object# is the currently
%                   measured object of type# which is coming from mask#.
%  featureNames     A cellarray with m entry describing the columns of
%                   relationFeatures.
%                   
%

mask1 = m1.Mask;
mask1Centers = m1.Centers;
object1MaxRadius = m1.MaxRadius;
mask1PixelChains = m1.PixelChains;

mask2 = m2.Mask;
mask2Centers = m2.Centers;
object2MaxRadius = m2.MaxRadius;
mask2PixelChains = m2.PixelChains;


%generate the reference image (to which we'll compare the objects on the
%other mask)
mergedMask2 = logical(sum(mask2,3));

nofObj1 = size(mask1Centers,1);
nofObj2 = size(mask2Centers,1);

nofOutputs = 6;

featureNames = cell(1,nofOutputs);
%init variables
relationFeatures = zeros(nofObj1,nofOutputs);
currentFeature = 0;

%1. OVERLAP INDEX
currentFeature = currentFeature + 1;
featureNames{currentFeature} = 'OverlapIndex';
%iterate through all layers
for i=1:size(mask1,3)          
   %if this is a single layer image we take advantage about the consecutive
   %numbering
   if size(mask1,3) ~= 1
       thisLayer = mask1(:,:,i);
       indicesInThisLayer = unique(thisLayer);
   else
       thisLayer = mask1;
       indicesInThisLayer = 1:nofObj1;
   end   
   
   if ~isempty(indicesInThisLayer)
       
        if (indicesInThisLayer(1) == 0) %get rid of non-object regions.
           indicesInThisLayer = indicesInThisLayer(2:end);
        end
       
        rExamined = regionprops(thisLayer,'Area');
        rExamined = cat(1,rExamined.Area);
        rOverlap = regionprops(thisLayer .* mergedMask2,'Area');
        rOverlapMatrix = zeros(size(rExamined));
        rOverlap = cat(1,rOverlap.Area);        
        rOverlapMatrix(1:length(rOverlap)) = rOverlap; %fill in the first elements.
        relationFeatures(:,currentFeature) = rOverlapMatrix ./ rExamined;
   end
   
end

%
%CUSTOM REMOVE FOR FASTENING

%2. MIXTURE INDEX
currentFeature = currentFeature + 1;
featureNames{currentFeature} = 'MixtureIndex';
featureNames{currentFeature+1} = 'ClosestDistance';
featureNames{currentFeature+2} = 'ClosestContourDistance';
featureNames{currentFeature+3} = 'MeanContourDistance';

if ~isempty(mask1Centers) && ~isempty(mask2Centers)
    %remove the layer information (if exist)
    mask1Centers = mask1Centers(:,1:2);
    mask2Centers = mask2Centers(:,1:2);

    distM = squareform(pdist([mask1Centers;mask2Centers]));
    %check for too big k
    if k>size(distM,1)-1
        k = size(distM,1)-1;
    end        

    %if there is at least 2 object.
    if k>0
        for i=1:nofObj1
            [~,nearestNeighbours] = sort(distM(i,:));
            nearestNeighbours = nearestNeighbours(2:end);
            relationFeatures(i,currentFeature) = sum(  nearestNeighbours(1:k)>nofObj1  )/k;
        end
    end
    
    
    %Calc. the closest distance to the related object
    %Do this inside, we can only calculate distances if we have at least
    %one object    
    
    %cross distance matrix row stands for the mask1
    crossDistM = distM(1:nofObj1,nofObj1+1:end);
    
    currentFeature = currentFeature + 1;    
    relationFeatures(:,currentFeature) = min(crossDistM,[],2);        
    
    %subtract the maximal possible radii from the cross distance matrix. In
    %the updated form the distances are lower bounds for the possible
    %minimal distances.
    updatedCrossDistM = crossDistM - repmat(convert2ColumnVector(object1MaxRadius),1,nofObj2);
    updatedCrossDistM = updatedCrossDistM - repmat(convert2RowVector(object2MaxRadius),nofObj1,1);
    
    [closestDistances,meanClosestDistances] = measureDistanceBetweenChains(updatedCrossDistM,mask1PixelChains,mask2PixelChains);
    
    currentFeature = currentFeature + 1;    
    relationFeatures(:,currentFeature) = closestDistances;
    
    currentFeature = currentFeature + 1;   
    relationFeatures(:,currentFeature) = meanClosestDistances;
    
else %SPECIAL CASES
    %if we came to the else branch therefore either mask1 or mask2 is empty
    %mixture index
    relationFeatures(1:nofObj1,currentFeature) = 0; % as if mask1 is empty then nofObjects = 0 so that it doesn't matter. Of mask2 is empty then there isn't any 'other typed' object therefore mixture index is 0.    
    %ClosestDistance to other typed object
    currentFeature = currentFeature + 1;
    relationFeatures(1:nofObj1,currentFeature) = Inf; %ifs similarly to above, if there isn't other typed object then distance from it is infinite.
    %ClosestContourDistance
    currentFeature = currentFeature + 1;
    relationFeatures(1:nofObj1,currentFeature) = Inf;
    %MeanContourDistance
    currentFeature = currentFeature + 1;
    relationFeatures(1:nofObj1,currentFeature) = Inf;
end

%}

currentFeature = currentFeature + 1;
featureNames{currentFeature} = 'NofRelatedObjects';
relationFeatures(:,currentFeature) = nofObj2;



end

%MeasureDistanceBetweenChains:
%   This functions computes the distance between 2 contours (pixel chains).
%   For each contour in set1 (mask1PixelChains) it outputs what is the
%   closest distance from any contour points in set2. (mask2PixelChains).
%   To optimize the process we take in a matrix which should give a lower
%   bound for the actual distance.
%   
%   INPUTS:
%       updatedCrossDistM: this is a distance matrix which should give a
%       lower bound for the actual distances between the pixelchains. It
%       has N1 rows and N2 columns. (N1 x N2 matrix)
%       mask1PixelChains: a cellarray with length N1 each entry contains an
%       n by 2 array which is the set of contour coordinates.
%       mask2PixelChains: same for set 2 its length is N2
%
%   OUTPUTS:
%       closestDistance: a vector with length N1, for each contour in set1
%       it says how far is the closest point from any contours in set2.
%       meanClosestDistance: a vector with length N1. For each contour in
%       set1: for each individual contour point measure what is the closest
%       set2 type contour point and take the mean of this distance.
%       (meanContourDistance)
function [closestDistance,meanClosestDistance] = measureDistanceBetweenChains(updatedCrossDistM,mask1PixelChains,mask2PixelChains)
    nofObj1 = length(mask1PixelChains);
    nofObj2 = length(mask2PixelChains);
    closestDistance = zeros(1,nofObj1);
    meanClosestDistance = zeros(1,nofObj1);
    
    for i=1:nofObj1        
        currentIndex = 1;
        [sortedDistValues,sortedDistIndices] = sort(updatedCrossDistM(i,:));                
        bigDistanceMatrix = pdist2(mask1PixelChains{i},mask2PixelChains{sortedDistIndices(currentIndex)});
        currentClosestDistances = min(bigDistanceMatrix,[],2);
        while (currentIndex<nofObj2 && max(currentClosestDistances)>sortedDistValues(currentIndex+1))
           currentIndex = currentIndex + 1;
           bigDistanceMatrix = pdist2(mask1PixelChains{i},mask2PixelChains{sortedDistIndices(currentIndex)});
           currentClosestDistances = min(currentClosestDistances,min(bigDistanceMatrix,[],2));
        end
        closestDistance(i) = min(currentClosestDistances);
        meanClosestDistance(i) = mean(currentClosestDistances);
    end
end
