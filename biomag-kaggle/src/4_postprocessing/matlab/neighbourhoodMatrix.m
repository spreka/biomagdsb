function nMtx = neighbourhoodMatrix(inImage, conn)
%neighbourhoodMatrix returns an n-by-n matrix nMtx of ones and zeros. n is the
% maximal object label in the image. nMtx(i,j) is set to 1 if they are
% adjacent.

uniqueValues = unique(inImage);
uniqueValues(uniqueValues==0) = [];
cellNumber = max(uniqueValues);

nMtx = zeros(cellNumber);

if ~isempty(nMtx)
    if conn==4
        se = [0 1 0; 1 1 1; 0 1 0];
    else
        se = [1 1 1; 1 1 1; 1 1 1];
    end

    for i=uniqueValues'
        currentCellImage = inImage == i;
        ringImage = imdilate(currentCellImage, se) - currentCellImage;
        neighbourPix = inImage(ringImage==1);
        neighbours = unique(neighbourPix);
        neighbours(neighbours==0) = [];
        nMtx(i, neighbours) = 1;
    end
end
