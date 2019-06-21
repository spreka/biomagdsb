function img = tetrisImage( maskArray, overlapProportion, imgSize, dist,nofCells)
%tetrisImage
%   Creates a joined image from the maskArray single mask images.
%   overlapProportion of the cells are allowed to be touching with others
%   overlap indicates touching
%
%   Modified to be more realistic: Remove the basement created by the
%   control points and the polyfit. Instead use alternating drop to random
%   locations and the tetris functionality. Additionally rotate the image
%   after a specified number of masks

blocks = [10,30,30];
%the blocks are specifying how many masks to place in one block. One block
%consists of first placing cells randomly then to tetris the remaining
%masks in the block. The ratio of randomly placed and tetrised masks are
%given in the next variable. If we ran out of the block size the process
%starts again from the beginning.
tetrisRatio = [0,0.75,0.75];

rotationBlock = 3;
%after rotationBlock number of masks the image is rotated by 90 degrees.

img = zeros(imgSize);
H = imgSize(1);

%controlY = rand(1,4)*H/5;
%controlX = round([1 imgSize(2)/3 imgSize(2)/3*2 imgSize(2)]);
%p = polyfit(controlX,controlY,3);
%topRow = round(polyval(p,1:imgSize(2)));
% the last position where we still have cell
topRow = zeros(1,imgSize(2));

%maximize inner distance with H/5;
dist = min(dist,round(H/5));

i = 1;
full = false;
imgRotCorrect = 1;
blockID = 1;
withinBlockIdx = 1;
while i<=length(maskArray) && ~full && i<=nofCells
    %fetch mask
    mask = maskArray{i};
    
    
    %do while to rotate if the image seems to be full from this direction    
    
    %BEGIN stem
    %check mode
    if withinBlockIdx < (1-tetrisRatio(blockID))*blocks(blockID)        
        [i,img,topRow,success] = placeInRandomMode(mask,imgSize,img,i,topRow);
        if ~success %if the mask could not be placed randomly, then try to tetris it
            [i,img,topRow,full] = placeInTetrisMode(mask,imgSize,i,topRow,H,img,overlapProportion,dist,full);
        end
    else
        [i,img,topRow,full] = placeInTetrisMode(mask,imgSize,i,topRow,H,img,overlapProportion,dist,full);
    end
    %END stem
    rotationState = 1;
    while full && rotationState < 4
        img = imrotate(img,90);
        topRow = calcTopRow(img);
        imgRotCorrect = mod(imgRotCorrect+1,2);        
        rotationState = rotationState + 1;
        %BEGIN stem
        %check mode
        if withinBlockIdx < tetrisRatio(blockID)*blocks(blockID)        
            [i,img,topRow,success] = placeInRandomMode(mask,imgSize,img,i,topRow);
            if ~success %if the mask could not be placed randomly, then try to tetris it
                [i,img,topRow,full] = placeInTetrisMode(mask,imgSize,i,topRow,H,img,overlapProportion,dist,full);
            end
        else
            [i,img,topRow,full] = placeInTetrisMode(mask,imgSize,i,topRow,H,img,overlapProportion,dist,full);
        end
        %END stem                    
    end
    
    
    if mod(i,rotationBlock) == 0
       img = imrotate(img,90);
       topRow = calcTopRow(img);
       imgRotCorrect = mod(imgRotCorrect+1,2);
    end
    
    %check if we still in this block
    if withinBlockIdx > blocks(blockID)                    
        if blockID<length(blocks)
            blockID = blockID + 1;
        else
            blockID = 1;
        end
        withinBlockIdx = 1;
    else
        withinBlockIdx = withinBlockIdx + 1;
    end
end

while imgRotCorrect ~=1
    img = imrotate(img,90);
    %    topRow = calcTopRow(img); it is not needed in this case
    imgRotCorrect = mod(imgRotCorrect+1,2);
end

%safety crop
img(H+1:end,:) = [];
img(:,imgSize(2)+1:end) = [];

end

function [i,img,topRow,success] = placeInRandomMode(mask,imgSize,img,i,topRow)
    nofTrials = 50;
    success = false;
    j = 1;
    [h,w] = size(mask);
    while ~success && j<nofTrials
        tryMask = zeros(size(img));
        randR = randi(imgSize(1)-h);
        randC = randi(imgSize(2)-w);
        tryMask(randR:randR+h-1,randC:randC+w-1) = mask;
        if ~any(tryMask & img)
            success = true;
        end
        j = j + 1;
    end
    if success
        tryMask(tryMask>0) = i;
        img = img + tryMask;
        currentMaskTops = zeros(1,w);
        for k = 1:w
            currentMaskTops(k) = find(tryMask(:,randC+k-1),1,'first');
        end
        topRow(randC:randC+w-1) = max(topRow(randC:randC+w-1),currentMaskTops);
        i = i + 1;
    end
end

function [i,img,topRow,full] = placeInTetrisMode(mask,imgSize,i,topRow,H,img,overlapProportion,dist,full)
    h = size(mask,1);
    w = size(mask,2);
    if imgSize(2)<w %if the mask is greater than the image
        i = i+1;
        return;
    end
    %ensure to put cell over an other one
    higherBottoms = find(topRow(1:end-w+1)>0);
    if ~isempty(higherBottoms)
        candStart = randi(length(higherBottoms));    
        c = higherBottoms(candStart);
    else
        c = randi(imgSize(2)-w);
    end
    approxBottomTouch = max(topRow(c:c+w-1))+1;
    if H-approxBottomTouch<h
        %search here the best fitting w sized area (where the max of the w wide block is minimal)
        maxiMin = max(topRow(1:w-1));
        maxiMinStart = 1;
        for j=2:imgSize(2)-w
            currMax = max(topRow(j:j+w-1));
            if currMax < maxiMin
                maxiMin = currMax;
                maxiMinStart = j;
            end
        end
        c = maxiMinStart;
        approxBottomTouch = maxiMin+1;
    end    
    if H-approxBottomTouch<h
        full = true;
    end
    if ~full
       [img,topRow] = placeMask(mask,img,c,approxBottomTouch,overlapProportion,i,topRow,dist);
       i = i+1;
    end       
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

function topRow = calcTopRow(img)
%less efective than updating constantly

H = size(img,1);
topRow = zeros(1,size(img,2));
for i=1:size(img,2)
    trcand = find(img(:,i),1,'first');
    if ~isempty(trcand), topRow(i) = H-trcand; end
end


end

