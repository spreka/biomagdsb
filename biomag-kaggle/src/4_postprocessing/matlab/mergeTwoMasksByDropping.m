function outImg = mergeTwoMasksByDropping(in1, in2, overlapThresh)

[median_size1, ~] = estimateCellSize2(in1);
[median_size2, ~] = estimateCellSize2(in2);

if ~isnan(median_size1) && ~isnan(median_size2)
    
    jointSize = ((median_size1+median_size2)/4)^2*pi;
    
    rstat1 = regionprops(in1, 'Area');
    rstat2 = regionprops(in2, 'Area');
    
    areas = [[rstat1.Area] [rstat2.Area]];
    
    imgChoiseIdxThresh = numel([rstat1.Area]) + 1;
    idxs = [1:numel([rstat1.Area]), (1:numel([rstat2.Area])) + imgChoiseIdxThresh];
    
    areaDiff = abs(areas - jointSize);
    
    [sortedAreas, idx] = sort(areaDiff);
    
    occupied = zeros(size(in1));
    outLabeled = zeros(size(in1));
    
    for i=1:numel(sortedAreas)
        
        current = idxs(idx(i));
        if current > imgChoiseIdxThresh
            currentPixs = find(in2 == (current - imgChoiseIdxThresh) );
        else
            currentPixs = find(in1 == current);
        end
        
        occupiedPix = sum(occupied(currentPixs));
        
        if occupiedPix / length(currentPixs) < overlapThresh        
            occupied(currentPixs) = 1;
            outLabeled(currentPixs) = current;
        end
        
    end
    
    outImg = relabelImage(outLabeled);
    
else
    
    if isnan(median_size1) && isnan(median_size2)
        outImg = uint16(zeros(size(in1)));
    elseif isnan(median_size1)
        outImg = in2;
    else
        outImg = in1;
    end
end
