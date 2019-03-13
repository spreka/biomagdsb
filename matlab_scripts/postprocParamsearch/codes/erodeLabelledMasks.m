function eroredLabels = erodeLabelledMasks(labelledMask, erosionRadius)

[~, labels] = bwdist( labelledMask>0 );

if max(labels(:)) == 0,
    labels = ones(size(labels));
end

erodedBW = imerode(labelledMask>0, strel('disk',erosionRadius));

eroredLabels = labelledMask(labels); 

eroredLabels(~erodedBW) = 0;
