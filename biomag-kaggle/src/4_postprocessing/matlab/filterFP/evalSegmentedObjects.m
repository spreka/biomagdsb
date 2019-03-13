function [TPs, FNs, FPs] = evalSegmentedObjects(gtMask, predMask)

thresholdLevels = 0.5:0.05:0.95;
nofThresh = length(thresholdLevels);

uvGT = unique(gtMask);
uvGT(uvGT==0) = [];
uvPM = unique(predMask);
uvPM(uvPM==0) = [];

nofGT = length(uvGT);
nofAllPM = length(uvPM);

TPs = zeros(nofGT,1);
FPs = ones(nofAllPM,1);

gtProps = regionprops(gtMask,'PixelIdxList');
pmProps = regionprops(predMask,'PixelIdxList');
gtIoUs = zeros(1,nofGT);

for gtInd=1:nofGT
    gtPixelIdxList = gtProps(uvGT(gtInd)).PixelIdxList;
    
    % collect ids of masks that overlap with current gt object
    overlappingPM = predMask( gtPixelIdxList );
    uvOverlappingPM = unique( overlappingPM );
    
    if uvOverlappingPM(1) == 0
        uvOverlappingPM(1) = [];
    end
    nofPM = length(uvOverlappingPM);
    
    for pmInd=1:nofPM
        
        IntersectionoverUnioun = IoU(gtPixelIdxList,pmProps(uvOverlappingPM(pmInd)).PixelIdxList);
        
        if IntersectionoverUnioun>gtIoUs(gtInd)
            gtIoUs(gtInd) = IntersectionoverUnioun;
        end
        
        
        if IntersectionoverUnioun>0.5
            TPs(gtInd) = 1;
            FPs(uvOverlappingPM(pmInd)) = 0;
        end
        
        
    end
end

FNs = ones(size(TPs))-TPs;

TP = sum(TPs,2);
FN = nofGT-sum(TPs,2);
FP = nofAllPM-TP;

end


function IoU = IoU(gtPixelIdxList,pmPixelIdxList)

intersectionArea = numel( find(ismember(gtPixelIdxList,pmPixelIdxList)) );
IoU = intersectionArea / (numel(gtPixelIdxList)+numel(pmPixelIdxList)-intersectionArea);

end
