function [object1MaxRadius,mask1PixelChains] = extractPixelChainsFromMask(mask1,mask1Centers,nofObj1)
%this function extracts the pixel chains (with their absolute coordinate in
%the image coordinate system. (axes according to regionprops)). It also
%calculates how far is the furthest perimeter point from the centroid

    object1MaxRadius = zeros(nofObj1,1);    
%    mask1PixelChains = cell(nofObj1,1);    

    %str = strel('square',3);
    %contourMask = mask1 - imerode(mask1,str);
    %pixelChainInStruct = regionprops(contourMask,'PixelList');
    filledPixelList = regionprops(mask1,'PixelList');
    mask1PixelChains = regionContour(filledPixelList); %drops away the 3rd dimension if exist from layers
    for i=1:nofObj1
        %mask1Contours = imdilate(mask1 == i,str) - (mask1==i);        
        %mask1PixelChains{i} = pixelChainInStruct(i).PixelList(:,1:2); %drop away the 3rd dimension if exists from the layers
        object1MaxRadius(i) = max(pdist2(mask1Centers(i,:),mask1PixelChains{i}));
    end
end

%The filled pixels is a cellarray given back by regionprops called with
%PixelList. Each entry is a matrix with 2 or 3 columns depending if our
%image is single layered (grayscale) or multichannel
%output a cellarray exactly as long as filledPixels that for each entry
%stores only the perimeter pixels in 2D.
function contourPixels = regionContour(filledPixels)
    contourPixels = cell(1,length(filledPixels));
    for i=1:length(filledPixels)
        actMatrix = filledPixels(i).PixelList;
        contourList = [];
        xStrech = unique(actMatrix(:,1));
        yStrech = unique(actMatrix(:,2));
        for j=1:length(xStrech)
            yCand = actMatrix(actMatrix(:,1) == xStrech(j),2);
            contourList = [contourList; xStrech(j) min(yCand); xStrech(j) max(yCand)];
        end
        for j=1:length(yStrech)
            xCand = actMatrix(actMatrix(:,2) == yStrech(j),1);
            contourList = [contourList; min(xCand) yStrech(j); max(xCand) yStrech(j)];
        end
        contourPixels{i} = contourList;
    end
end