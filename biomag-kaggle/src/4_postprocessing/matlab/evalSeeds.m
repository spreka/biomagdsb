function metrics = evalSeeds(gtMask,seeds)
%EVALSEEDS A function to evaluate seed detection compared to the ground truth. Several metrics are computed.
% metrics = evalSeeds(gtMask,seeds) Evaluates the precision of seed
% detection compared to ground truth data. metrics is a structure
% containing calculated values of accuracy.
%
% NOTE: All the seeds must be consecutively labeled from 1 to the
% maximum value in the image.
%
% Metrics calculated:
%
% TP        - number of GT objects that has overlapping at least one seed
% FN        - number of GT objects that has no overlapping seeds
% FP_IN     - number of seeds that are inside a ground truth object but it
%             is not the only hit
% FP_OUT    - number of seeds over the background (real false positives)

uvGT = unique(gtMask);
uvPM = unique(seeds);

nofGT = length(uvGT)-1;
nofAllSeeds = length(uvPM)-1;

TPs = zeros(nofGT,1);
% FNs = zeros(nofGT,1);
% FP_INs = zeros(nofAllSeeds,1);
% FP_OUTs = zeros(nofAllSeeds,1);
FP_IN = 0;
% FP_OUT = 0;

gtProps = regionprops(gtMask, 'Centroid', 'PixelIdxList');
% pProps = regionprops(seeds, 'PixelIdxList');

% seedIdxList = find(seeds>0);
% [seedCentroidsX, seedCentroidsY] = ind2sub(size(seeds),seedIdxList);

% dists = pdist2(cat(1,gtProps.Centroid), [seedCentroidsX seedCentroidsY]);

for gtInd=1:nofGT
    gtPixelIdxList = gtProps(uvGT(gtInd+1)).PixelIdxList;
    
    % collect ids of seeds that overlap with current gt object
    overlappingSeeds = seeds( gtPixelIdxList );
    uvOverlappingSeeds = unique( overlappingSeeds );
    
    if uvOverlappingSeeds(1) == 0
        uvOverlappingSeeds(1) = [];
    end
    nofS = length(uvOverlappingSeeds);
    
    if nofS>0
        TPs(gtInd) = 1;
        FP_IN = FP_IN + nofS - 1;
    end

end



TP = sum(TPs);
FN = nofGT-TP;
FP_OUT = nofAllSeeds - TP - FP_IN;
% FP = nofAllSeeds-TP;
% 
% precision = mean(TP./(TP+FP+FN));

metrics = struct('TP', TP, 'FN', FN, 'FP_IN', FP_IN, 'FP_OUT', FP_OUT);

end

