function inImg = removeObjectWithinObject(inImg)

cellNumber = max(inImg(:));

for i=1:cellNumber
   
    currCellImg = inImg == i;
    
    
    currCellImgFilled = imfill(currCellImg ,'holes');
    pix = find(currCellImgFilled > 0);
    inImg(pix) = i;
end

inImg = relabel(inImg);
