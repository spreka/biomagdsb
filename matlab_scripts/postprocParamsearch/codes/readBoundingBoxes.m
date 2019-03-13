function bboxes = readBoundingBoxes(bbFileName)

importedData = importdata(bbFileName);

% bboxes = dlmread(bbFileName);

if isstruct(importedData)
    bboxes = importedData.data;
else
    bboxes = importedData;
end

emptyRowIdx = zeros(size(bboxes,1),1);

for i=1:size(bboxes,1)
    if ~any(bboxes(i,:)>0)
        emptyRowIdx(i) = 1;
    end
end

bboxes( find(emptyRowIdx),:) = [];
