function generateMasksToSplittedClusters(splittedDir,targetDir,objType)
%copies to the style transfer input folder
%both real images and estimated masks has the same format, below their root
%there are the clusters

%doc: the folder structure of splittedDir is:
%split0 -> p2ptrain -> cluster1 (duplicated images)
%                   -> cluster2
%       -> inputClusters -> cluster1 (simple images)
%                        -> cluster2
%split1 -> etc.

d = dir(splittedDir);

%for each split
for i=3:numel(d)
    generateMasksForCluster(fullfile(splittedDir,d(i).name,'preds-clusters'), fullfile(targetDir,d(i).name,'generated'), '.tiff', 20, objType);
end


end