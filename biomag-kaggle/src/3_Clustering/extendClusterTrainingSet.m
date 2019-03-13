function extendClusterTrainingSet(testDir,trainDir,trainStyleCsv,predStyleCsv,fileList)
%extends the clusterTrainingSet
%   testDir is the directory from which to extend the training set
%   trainDir is the directory to which we copy the data
%   If in both case we have the feature database, then they are copied
%   simultaneously.
%   trainStyleCsv, is a csv file that contians the style IDs used for
%   training
%   predStyleCsv contains entries for all elements in fileList, and their
%   corresponding image cluster.

predT = readtable(predStyleCsv);
trainT = readtable(trainStyleCsv);

prevMaxCluster = max(trainT.Style);

moveImagesWithFeatures( testDir,trainDir, fileList );

moveIndex = zeros(length(predT.Name),1);
for i=1:length(predT.Name)
    if any(1-cellfun(@isempty,strfind(fileList,predT.Name{i})))
        moveIndex(i) = 1;
    end
end
moveIndex = logical(moveIndex);
toMoveTable = predT(moveIndex,:);
indices = unique(toMoveTable.Style);
sequentialClusterIndices = zeros(length(toMoveTable.Style),1);
for i=1:length(toMoveTable.Style)
    sequentialClusterIndices(i) = prevMaxCluster + find(indices == toMoveTable.Style(i));
end
toMoveTable.Style = sequentialClusterIndices;
trainT = [trainT; toMoveTable];
%predT(moveIndex,:) = [];

writetable(trainT,trainStyleCsv);
%writetable(predT,predStyleCsv);



end

