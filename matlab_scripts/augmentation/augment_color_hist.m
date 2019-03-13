function outImg = augment_color_hist(outImg, histStretchProb, imStretch, histEq)
% inImg - 24 bit image
%
% imStretch - stretching the image with certain tolerance 0.01 recommended,
% histEq  - histogram equalizaiton
% noise
%
% Peter Horvath 2018

% stretch intensities
if histStretchProb
    outImg = uint8(outImg);
    outImg(:,:,1) = imadjust(outImg(:,:,1),stretchlim( outImg(:,:,1), imStretch));
    outImg(:,:,2) = imadjust(outImg(:,:,2),stretchlim( outImg(:,:,2), imStretch));
    outImg(:,:,3) = imadjust(outImg(:,:,3),stretchlim( outImg(:,:,3), imStretch));
end

% histogram equalization
if histEq
    outImg = uint8(outImg);
    outImg(:,:,1) = histeq(outImg(:,:,1));
    outImg(:,:,2) = histeq(outImg(:,:,2));
    outImg(:,:,3) = histeq(outImg(:,:,3));
end

% add noise





outImg = uint8(outImg);

