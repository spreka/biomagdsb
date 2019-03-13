function correctedImage = correctWithUnet(inSegm, probMap, probThresh, erRad, dilRad, minOverlap, maxVParam, cAreaParam)
%tunes segmentation boundary with the help of UNet probabilities. erRad and
%dilRad is the maximal allowed distance of changing object contours inwards and
%outwards respectively.

inSegm = double(inSegm);
probMap = double(probMap > probThresh);

maskSmall = double(erodeLabelledMasks(inSegm, erRad));
maskBig = double(dilateLabelledMasks(inSegm, dilRad));

ring = double(maskBig-maskSmall) .* probMap;

out = ring + maskSmall;
outFinal = zeros(size(out));

index = 1;
for j=1:max(out(:))
    blank = zeros(size(out));
    blank(out == j) = 1;
    labelledBlank = bwlabel(blank, 4);
    stats = regionprops(blank, 'Area');
    if ~isempty(stats)
        [maxv, maxi] = max(stats.Area);
        if maxv > maxVParam
            outFinal(labelledBlank == maxi) = index;
            index = index + 1;
        end
    end
end

%% remove false positives based on UNet
index = 1;
outNoFPos = zeros(size(out));

for j=1:max(outFinal(:))
    pix = find(outFinal == j);
    cArea = numel(pix);
    probMapSum = sum(probMap(pix));
    if probMapSum / cArea >= minOverlap
        if cArea>=cAreaParam
            outNoFPos(pix) = index;
            index = index + 1;
        end
    end
end

correctedImage = uint16(outNoFPos);
