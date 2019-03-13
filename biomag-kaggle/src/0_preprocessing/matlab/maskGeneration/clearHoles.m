function filteredMaskArray = clearHoles(maskArray)
%maskarray is a cellarray with mask images and we first try to correct
%wrong masks by filling them out, and then if it is still not a full object
%then delete it

j = 1;
N = numel(maskArray);
filteredMaskArray = cell(1,N);
for i=1:N
    if ~(bweuler(maskArray{i},4) == 1 && bweuler(maskArray{i},8) == 1)
        tmpMask = imfill(maskArray{i},'holes');
        if bweuler(tmpMask,4) == 1 && bweuler(tmpMask,8) == 1
            filteredMaskArray{j} = tmpMask;
            j = j+1;
        end
    else
        filteredMaskArray{j} = maskArray{i};
        j = j+1;
    end
end

filteredMaskArray(j:end) = [];

end