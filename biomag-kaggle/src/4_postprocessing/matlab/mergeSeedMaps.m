function seedMap = mergeSeedMaps(seeds, varargin)
% seeds is an mxnxt matrix, where m and n the size of input image, and t is
% the number of different seed detection methods
% second argument can be the bandwith parameter of mean shift algorithm

% radius for mean calculation in pixels, can be get from object size histogram
bWidth = 20; 
if ~isempty(varargin)
    bWidth = varargin{1};
end

mergedSeedMap = any(seeds,3);

[seedCoordsX, seedCoordsY] = ind2sub(size(mergedSeedMap), find(mergedSeedMap));

[clustCent,point2cluster,clustMembsCell] = HGMeanShiftCluster([seedCoordsX, seedCoordsY]', bWidth, 'flat', 0);

seedMap = zeros(size(mergedSeedMap));
seedMap (sub2ind(size(seedMap), int16(clustCent(1,:)), int16(clustCent(2,:)) )) = 1;
seedMap = bwlabel(seedMap);
