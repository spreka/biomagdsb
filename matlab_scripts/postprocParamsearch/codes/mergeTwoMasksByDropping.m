function outImg = mergeTwoMasksByDropping(in1, in2, overlapThresh, areaThresh)

[median_size1, std_size1] = estimateCellSize2(in1);
[median_size2, std_size2] = estimateCellSize2(in2);

jointSize = ((median_size1+median_size2)/4)^2*pi;

rstat1 = regionprops(in1, 'Area');
rstat2 = regionprops(in2, 'Area');

if (isempty(rstat1) == true && isempty(rstat2) == false) 
    outImg = relabelImage(in2); return;
elseif (isempty(rstat1) == false && isempty(rstat2) == true) 
    outImg = relabelImage(in1); return;
elseif (isempty(rstat1) == false && isempty(rstat2) == false)
    outImg = relabelImage(in1); return;
end     

areas = [[rstat1.Area] [rstat2.Area]];

idxs = [1:numel([rstat1.Area]), (1:numel([rstat2.Area])) + areaThresh];


areaDiff = abs(areas - jointSize);

[sortedAreas, idx] = sort(areaDiff);

occupied = in1 * 0;
outLabeled = in1 * 0;

for i=1:numel(sortedAreas)
    
    current = idxs(idx(i));
    % TODO change thresh?
    if current > areaThresh
        currentPixs = find(in2 == current - areaThresh);
    else
        currentPixs = find(in1 == current);
    end

    occupiedPix = sum(occupied(currentPixs));
        
    % TODO change thresh?
    if occupiedPix / length(currentPixs) < overlapThresh
        
        occupied(currentPixs) = 1;
        outLabeled(currentPixs) = current;
        
    end
    
end
    
outImg = relabelImage(outLabeled);
