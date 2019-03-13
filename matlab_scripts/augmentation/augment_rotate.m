function outImg = augment_rotate(outImg, rotateNo)

% rotation
[sx, sy, sz] = size(outImg);

if rotateNo ~= 0
    if sz > 1
        rI = rot90(outImg(:,:,1), rotateNo);
        gI = rot90(outImg(:,:,2), rotateNo);
        bI = rot90(outImg(:,:,3), rotateNo);
        outImg = []; outImg(:,:,1) = rI; outImg(:,:,2) = gI; outImg(:,:,3) = bI;
        outImg = uint8(outImg);
    else
        outImg = rot90(outImg, rotateNo);
        outImg = uint16(outImg);
    end
end
