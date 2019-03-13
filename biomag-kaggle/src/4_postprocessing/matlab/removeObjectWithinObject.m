function inImg = removeObjectWithinObject(inImg)
%removeObjectWithinObject vanishes embedded smaller objects via filling
%cavities of the container objects.

uniqueValues = unique(inImg);
uniqueValues(uniqueValues==0)=[];

if ~isempty(uniqueValues)
    uniqueValues = reshape(uniqueValues, 1, []);
    for i=uniqueValues
        currCellImg = inImg == i;
        currCellImgFilled = imfill(currCellImg ,'holes');
        inImg(currCellImgFilled > 0) = i;
    end
    
end

inImg = relabelImage(inImg);
