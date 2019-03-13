function augmentTrainingSet(inFolder, outFolder, styleData, multiplyAugment)
%%%
% This script generates a training set such that the median cell size is
% fix and crops into images of a fix size and applies random augmentation

expectedCellSize = 40; % this is the diameter 40->256
expectedCropSize = 512;

if styleData == 0
    % augmentaiton for real images
    numAugment = 1;
    histEqPorob = 0.05; %0.05;
    histStretchProb = 0.1; %0.2;
    histStretchLevel = 0.03; %0.04;
    invert = 0.1;
    blurProb = 0.2;
    blurLevel = 2;
    noiseProb = 0.2;   
    colorFlip = 0;%0.1;
    maskFormat = '.tiff';
    stepSize = 1;    
elseif styleData == 1
    % augmentaiton for style transfer
    numAugment = 1;
    histEqPorob = 0.05;
    histStretchProb = 0.1;
    histStretchLevel = 0.02;
    invert = 0.0;
    blurProb = 0.2;
    blurLevel = 4;
    noiseProb = 0.2;       
    colorFlip = 0.0;
    maskFormat = '.tiff'; % TODO change back to .png
    stepSize = 6%10;
elseif styleData == 2
    % augmentaiton for validation RCNN
    numAugment = 1;
    histEqPorob = 0.0;
    histStretchProb = 0.0;
    histStretchLevel = 0.0;
    invert = 0.0;
    blurProb = 0.0;
    blurLevel = 0;
    noiseProb = 0.0;       
    colorFlip = 0.0;
    maskFormat = '.tiff'; % TODO change back to .png
    stepSize = 1;
end

if nargin == 4
    numAugment = multiplyAugment;
end

mkdir(outFolder);
mkdir([outFolder filesep 'images']);
mkdir([outFolder filesep 'masks']);

fileList = dir([inFolder filesep 'images' filesep '*.png']);

for i=1:stepSize:numel(fileList)
    disp(fileList(i).name);
    % read in the image and the mask
    inImg = imread([inFolder filesep 'images' filesep fileList(i).name ]);
    if size(inImg, 3) == 1
        inImg(:,:,2) = inImg(:,:,1);
        inImg(:,:,3) = inImg(:,:,1);
    end
        
    maskImg = uint16(imread([inFolder filesep 'masks' filesep fileList(i).name(1:end-4) maskFormat ]));
    
    % estimage cell size
    [median_size, std_size] = estimateCellSizeFromMask(maskImg);
    
    % resize orig image
    resizeFactor = expectedCellSize / median_size;
    resImage = imresize(inImg, resizeFactor, 'bicubic');
    resMask = imresize(maskImg, resizeFactor, 'nearest');
    
    [sx, sy] = size(resMask);
    
    % determine number of crops
    % case 1
    if sx <= expectedCropSize && sy <= expectedCropSize
        outImg = zeros(expectedCropSize, expectedCropSize, 3);
        outMask = zeros(expectedCropSize, expectedCropSize);
        resImageR = resImage(:,:,1); resImageG = resImage(:,:,2); resImageB = resImage(:,:,3);
        outImg(:,:,1) = median(resImageR(:));
        outImg(:,:,2) = median(resImageG(:));
        outImg(:,:,3) = median(resImageB(:));
        
        outImg(1:sx, 1:sy, :) =  resImage;
        outMask(1:sx, 1:sy) = resMask;
        for j=1:numAugment
            outName = [fileList(i).name(1:end-4) '_aug' randHexString(6)];
            [outImg, outMask] = augment_image(outImg, outMask, rand(1) < histStretchProb, rand(1) * histStretchLevel, rand(1) < histEqPorob, rand(1) < invert, rand(1) < colorFlip, rand(1) < noiseProb, randi(3), rand(1) < blurProb, rand(1) * blurLevel);            
            if max(outMask(:)) > 0
                % non empty !!
                imwrite(outImg, [outFolder filesep 'images' filesep outName '.png']);
                imwrite(uint16(outMask), [outFolder filesep 'masks' filesep outName '.tiff']);
            end
        end
        % case 2
    elseif sx <= expectedCropSize && sy > expectedCropSize
        % two random crops
        for j=1:2*numAugment
            x1 = 1;
            x2 = sx;
            y1 = randi(sy - expectedCropSize);
            y2 = y1+expectedCropSize - 1;
            outImg =  resImage(x1:x2, y1:y2, :);
            outMask = resMask(x1:x2, y1:y2);
            outMask = relabel(outMask);
            outName = [fileList(i).name(1:end-4) '_aug' randHexString(6)];
            [outImg, outMask] = augment_image(outImg, outMask, rand(1) < histStretchProb, rand(1) * histStretchLevel, rand(1) < histEqPorob, rand(1) < invert, rand(1) < colorFlip, rand(1) < noiseProb, randi(3), rand(1) < blurProb, rand(1) * blurLevel);                        
            if max(outMask(:)) > 0
                % non empty !!
                imwrite(outImg, [outFolder filesep 'images' filesep outName '.png']);
                imwrite(uint16(outMask), [outFolder filesep 'masks' filesep outName '.tiff']);
            end
        end
        % case 3
    elseif sx > expectedCropSize && sy <= expectedCropSize
        % two random crops
        for j=1:2*numAugment
            y1 = 1;
            y2 = sy;
            x1 = randi(sx - expectedCropSize);
            x2 = x1+expectedCropSize - 1;
            outImg =  resImage(x1:x2, y1:y2, :);
            outMask = resMask(x1:x2, y1:y2);
           outMask = relabel(outMask);
            outName = [fileList(i).name(1:end-4) '_aug' randHexString(6)];
            [outImg, outMask] = augment_image(outImg, outMask, rand(1) < histStretchProb, rand(1) * histStretchLevel, rand(1) < histEqPorob, rand(1) < invert, rand(1) < colorFlip, rand(1) < noiseProb, randi(3), rand(1) < blurProb, rand(1) * blurLevel);                        
            if max(outMask(:)) > 0
                % non empty !!
                imwrite(outImg, [outFolder filesep 'images' filesep outName '.png']);
                imwrite(uint16(outMask), [outFolder filesep 'masks' filesep outName '.tiff']);
            end
        end
        % case 4
    elseif sx > expectedCropSize && sy > expectedCropSize
        % number of random crops
        numCrop = round(sqrt((sx*sy) / expectedCropSize^2)) + 2;
        for j=1:2*numCrop*numAugment
            x1 = randi(sx - expectedCropSize);
            x2 = x1+expectedCropSize - 1;
            y1 = randi(sy - expectedCropSize);
            y2 = y1+expectedCropSize - 1;
            outImg =  resImage(x1:x2, y1:y2, :);
            outMask = resMask(x1:x2, y1:y2);
            outMask = relabel(outMask);
            outName = [fileList(i).name(1:end-4) '_aug' randHexString(6)];
            [outImg, outMask] = augment_image(outImg, outMask, rand(1) < histStretchProb, rand(1) * histStretchLevel, rand(1) < histEqPorob, rand(1) < invert, rand(1) < colorFlip, rand(1) < noiseProb, randi(3), rand(1) < blurProb, rand(1) * blurLevel);            
            if max(outMask(:)) > 0
                % non empty !!
                imwrite(outImg, [outFolder filesep 'images' filesep outName '.png']);
                imwrite(uint16(outMask), [outFolder filesep 'masks' filesep outName '.tiff']);
            end
        end
    end    
end
