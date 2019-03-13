function [median_size, std_size] = estimateCellSize2(segmentation)

props = regionprops(segmentation, 'EquivDiameter');
props( [props.EquivDiameter]==0 ) = [];
allSize = cat(1,props.EquivDiameter);

median_size = median(allSize);
std_size = std(allSize);
