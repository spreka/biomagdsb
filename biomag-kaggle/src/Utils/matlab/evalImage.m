function [precision,gtIoUs] = evalImage(gtMask,predMask)

thresholdLevels = 0.5:0.05:0.95;
nofThresh = length(thresholdLevels);

uvGT = unique(gtMask);
uvGT(uvGT==0) = [];
uvPM = unique(predMask);
uvPM(uvPM==0) = [];

nofGT = length(uvGT);
nofAllPM = length(uvPM);

TPs = zeros(nofThresh,nofGT);

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
        
        for tInd = 1:nofThresh
            thresh = thresholdLevels(tInd);
                        
            if IntersectionoverUnioun>thresh
                TPs(tInd,gtInd) = 1;                
            end
            
        end
    end
end

TP = sum(TPs,2);
FN = nofGT-sum(TPs,2);
FP = nofAllPM-TP;

precision = mean(TP./(TP+FP+FN));
end


function IoU = IoU(gtPixelIdxList,pmPixelIdxList)

intersectionArea = numel( find(ismember(gtPixelIdxList,pmPixelIdxList)) );
IoU = intersectionArea / (numel(gtPixelIdxList)+numel(pmPixelIdxList)-intersectionArea);

end
