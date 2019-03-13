function nMtx = neighbourhoodMatrix(inImage, conn)

cellNumber = max(inImage(:));

nMtx = zeros(cellNumber);

if conn==4
    se = [0 1 0; 1 1 1; 0 1 0];
else
    se = [1 1 1; 1 1 1; 1 1 1];
end

for i=1:cellNumber
    currentCell = inImage == i;
    % dilate 1 pixel
    currentCellImage = inImage * 0;
    currentCellImage(currentCell) = 1;    
    ringImage = imdilate(currentCellImage, se) - currentCellImage;
    ringPix = find(ringImage == 1);
    neighbourPix = inImage(ringPix);
    neighbours = unique(neighbourPix);
    for j=1:numel(neighbours)
        if neighbours(j) > 0
            nMtx(i, neighbours(j)) = 1;
        end
    end    
end