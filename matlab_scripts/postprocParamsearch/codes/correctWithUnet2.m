function correctedImgMap = correctWithUnet2(imagesMap, probsMap, probThresh, erRad, dilRad, minOverlap, maxVParam, cAreaThresh)
%tunes segmentation boundary with the help of UNet probabilities. erRad and
%dilRad is the maximal allowed distance of changing object contours inwards and
%outwards respectively. It also discards false positives based on UNet
%probability map. If an object has an overlap less then 65% of its area, it
%is considered as false positive.

correctedImgMap = containers.Map;
allKeys = imagesMap.keys();

for i=1:length(allKeys)
    
    inSegm = double(imagesMap(allKeys{i}));
                
    scaleUNET = 32;
    probMap = double(probsMap(allKeys{i}) >= (probThresh * (65535 / scaleUNET)));
    
    %probMap = double(imgaussfilt(probMapT, 2));
    
    maskSmall = double(erodeLabelledMasks(inSegm, erRad));
    maskBig = double(dilateLabelledMasks(inSegm, dilRad));
    
    ring = double(maskBig-maskSmall) .* probMap;

    out = ring + maskSmall;
    outFinal = out * 0;
    
    index = 1;
    for j=1:max(out(:))
        
        blank = out * 0;
        
        blank(out == j) = 1;
        labelledBlank = bwlabel(blank, 8);
        stats = regionprops(blank, 'Area');
        if ~isempty(stats)
            [maxv, maxi] = max(stats.Area);
            if maxv > maxVParam
                outFinal(labelledBlank == maxi) = index;                
                index = index + 1;
            end
        end
    end
    
    %imwrite(uint16(outFinal), fullfile('/home/baran/Devel/', strcat(allKeys{i}, '.png')));
        
    %% remove false positives
    index = 1;
    outNoFPos = out * 0;
     
    for j=1:max(outFinal(:))
        pix = find(outFinal == j);
        cArea = numel(pix);
        probMapSum = sum(probMap(pix));
        overlap = probMapSum / cArea;
        if overlap >= minOverlap
            if cArea >= cAreaThresh
                outNoFPos(pix) = index;
                index = index + 1;
            end
        else
            if cArea >= cAreaThresh * 4
                outNoFPos(pix) = index;
                index = index + 1;
            else
                fprintf('%i %f\n', cArea, overlap);
                disp([allKeys{i} ', Removed: ' num2str(j)]);
            end
        end
    end
        
    % Add false negati    
        
    correctedImgMap(allKeys{i}) = uint16(outNoFPos);
       
end
