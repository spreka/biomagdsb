function [median_size, std_size] = estimateCellSizeFromMask(maskImg)

props = regionprops(maskImg, 'EquivDiameter'); 
allSize = [props.EquivDiameter];    

median_size = median(allSize);
std_size = std(allSize);
