function seedMap = bbox2seeds(bboxes, imageSize)

seedMap = uint16(zeros(imageSize));

if ~isempty(bboxes)
    seedCoordsY = uint16((bboxes(:,3) - bboxes(:,1))/2 + bboxes(:,1));
    seedCoordsX = uint16((bboxes(:,4) - bboxes(:,2))/2 + bboxes(:,2));
    % max(seedCoordsX)
    % max(seedCoordsY)
    
    seedCoordsX(seedCoordsX>imageSize(1)) = imageSize(1);
    seedCoordsY(seedCoordsY>imageSize(2)) = imageSize(2);
    
    % max(seedCoordsX)
    % max(seedCoordsY)
    
    seedIdx = sub2ind(imageSize, seedCoordsX, seedCoordsY);
    
    seedMap(seedIdx) = 1;
    seedMap = bwlabel(seedMap);
end