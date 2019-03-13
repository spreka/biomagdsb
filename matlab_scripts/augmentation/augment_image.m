function [outImg, outMask] = augment_image(outImg, outMask, histStretchProb, imStretch, histEq, invert, colorFlip, addNoise, noise, addBlur, blur)

% inImg - 24 bit image
%
% imStretch - stretching the image with certain tolerance 0.01 recommended,
% histEq  - histogram equalizaiton
%
% Peter Horvath 2018

% stretch
outImg = augment_color_hist(outImg, histStretchProb, imStretch, histEq);

% rotate
rotateNo = randi(4);
outImg = augment_rotate(outImg, rotateNo);
outMask = augment_rotate(outMask, rotateNo);

% swap colors
outImg = augment_colorFlip(outImg, invert, colorFlip);

% blur
if addBlur
    outImg = augment_blur(outImg, blur);
end

% noise
if addNoise
    outImg = augment_noise(outImg, noise);    
end

outImg = uint8(outImg);

