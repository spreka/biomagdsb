function correctedImgMap = correctWithUnet2(imagesMap, probsMap, probThresh, erRad, dilRad, minOverlap, maxVParam, cAreaParam)
%tunes segmentation boundary with the help of UNet probabilities. erRad and
%dilRad is the maximal allowed distance of changing object contours inwards and
%outwards respectively. It also discards false positives based on UNet
%probability map. If an object has an overlap less then 65% of its area, it
%is considered as false positive.

correctedImgMap = containers.Map;
allKeys = probsMap.keys();

for i=1:length(allKeys)

    inSegm = double(imagesMap(allKeys{i}));
    probMap = double(probsMap(allKeys{i}) > probThresh);
    
%     inSegm = double(erodeLabelledMasks(inSegm, 1));
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
%                 if maxv > 630
%                     outFinal(mask == j) = index;
%                 else
                    outFinal(labelledBlank == maxi) = index;
%                 end
                
                index = index + 1;
            end
        end
    end
        
    %% remove false positives
    index = 1;
    outNoFPos = zeros(size(out));
%     minOverlap = 0.65;
     
    for j=1:max(outFinal(:))
        pix = find(outFinal == j);
        cArea = numel(pix);
        probMapSum = sum(probMap(pix));
        if probMapSum / cArea >= minOverlap
            if cArea>=cAreaParam
                outNoFPos(pix) = index;
                index = index + 1;
            end
        else
            disp([allKeys{i} ', Removed: ' num2str(j)]);
        end
    end
    
    correctedImgMap(allKeys{i}) = uint16(outNoFPos);
end
