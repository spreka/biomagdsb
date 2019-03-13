function maskArray = resizeMasks(maskArray,estimatedAreas)
%resizes all masks from maskArray to follow a normal distribution with
%respect to area defined by the mean of estimatedAreas and std of it.

meanA = mean(estimatedAreas);
stdA = std(estimatedAreas);

for i=1:numel(maskArray)
    A = regionprops(maskArray{i},'Area');
    currentA = normrnd(meanA,min(meanA*0.2,stdA)); %the standard deviation is maximized by the 20% of the average mean
    imresize(maskArray{i},currentA/A.Area,'nearest');
end

end