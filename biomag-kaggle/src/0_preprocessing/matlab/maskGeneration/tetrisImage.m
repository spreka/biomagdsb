function img = tetrisImage( maskArray, overlapProportion, imgSize, dist,nofCells)
%tetrisImage
%   Creates a joined image from the maskArray single mask images.
%   overlapProportion of the cells are allowed to be touching with others
%   overlap indicates touching

img = zeros(imgSize);
H = imgSize(1);

controlY = rand(1,4)*H/5;
controlX = round([1 imgSize(2)/3 imgSize(2)/3*2 imgSize(2)]);
p = polyfit(controlX,controlY,3);
topRow = round(polyval(p,1:imgSize(2)));
% the last position where we still have cell

%maximize inner distance with H/5;
dist = min(dist,round(H/5));

i = 1;
full = 0;
while i<=length(maskArray) && ~full && i<=nofCells
    mask = maskArray{i};
    try
        c = selectColumn(mask,imgSize);
    catch
        i = i+1;
        continue; % in the unlikely event that the mask is grater then the image
    end
    h = size(mask,1);
    w = size(mask,2);
    approxBottomTouch = max(topRow(c:c+w-1))+1;
    if H-approxBottomTouch<h
            [~,c] = min(topRow(1:end-w));
        approxBottomTouch = max(topRow(c:c+w-1))+1;
    end    
    if H-approxBottomTouch<h
        full = 1;
    end
    if ~full
       [img,topRow] = placeMask(mask,img,c,approxBottomTouch,overlapProportion,i,topRow,dist);
    end        
    i = i+1;
end

img = imtranslate(img,[0, -round(rand*(H-max(topRow)))]);

%safety crop
img(H+1:end,:) = [];
img(:,imgSize(2)+1:end) = [];

end

function c = selectColumn(mask,imgSize)
    maxSize = imgSize(2)-size(mask,2);
    c = randi(maxSize);
end

function [img,topRow] = placeMask(mask,img,c,bottomStart,oP,i,topRow,nonOverlapOffset)
    H = size(img,1);
    h = size(mask,1);
    w = size(mask,2);
    mS = zeros(1,w);%mask start
    mE = zeros(1,w);%mask end        
    for j=1:w
        if isempty(find(mask(:,j),1,'first'))
            mS(j) = 1;
        else
            mS(j) = find(mask(:,j),1,'first');
        end
        if isempty(find(mask(:,j),1,'last'))
            mE(j) = h;
        else            
            mE(j) = find(mask(:,j),1,'last');
        end
    end    
    if rand<oP       
       % if overlap then push it to the next one
       diff = bottomStart-topRow(c:c+w-1) + (h-mE);
       offset = min(diff);
       bottomStart = bottomStart - offset;
    else
       if H-bottomStart-h>nonOverlapOffset
            bottomStart = bottomStart + round(randi(nonOverlapOffset));
       end
    end
    mask(mask>0) = i;
    for j=1:w
        img(H-bottomStart-h+1+mS(j):H-bottomStart-h+1+mE(j),c+j-1) = mask(mS(j):mE(j),j);
    end
    topRow(c:c+w-1) = bottomStart + h - mS;   
    
    %imagesc(img);
    %pause(0.1);
        
end

