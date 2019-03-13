function outImg = mergeTwoFilesFINAL(in1, in2, scaleTh)

[median_size1, std_size1] = estimateCellSizeFromMask(in1);
[median_size2, std_size2] = estimateCellSizeFromMask(in2);


jointSize = ((median_size1+median_size2)/4)^2*pi;

rstat1 = regionprops(in1, 'Area');
rstat2 = regionprops(in2, 'Area');

areas = [[rstat1.Area] [rstat2.Area]];

idxs = [1:numel([rstat1.Area]), (1:numel([rstat2.Area])) + 10000];


areaDiff = abs(areas - jointSize);

[sortedAreas, idx] = sort(areaDiff);

occupied = in1 * 0;
outLabeled = in1 * 0;

for i=1:numel(sortedAreas)
    
    current = idxs(idx(i));
    if current > 10000
        currentPixs = find(in2 == current - 10000);
    else
        currentPixs = find(in1 == current);
    end

    occupiedPix = sum(occupied(currentPixs));
        
    
    if occupiedPix / length(currentPixs) < scaleTh
        
        occupied(currentPixs) = 1;
        outLabeled(currentPixs) = current;
        
    end
    
end
    
outImg = relabel(outLabeled);
