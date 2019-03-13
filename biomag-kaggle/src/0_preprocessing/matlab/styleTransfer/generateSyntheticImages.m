numOfOutputImages = 5;

coverThreshold = 0.2;
clustProb = 0.75;

segmentationDir = 'd:\Projects\Data Science Bowl 2018\data\__segmentation\RCNN_20180317_Hist_prediction_from_Kriston\test\';
segmentationExtension = '.png';

% gtDir = 'd:\Projects\Data Science Bowl 2018\data\__ground-truth\';
gtDir = 'd:\__ground-truth\';

rawImageDir = 'd:\Projects\Data Science Bowl 2018\data\__raw-images\out_stage1_test_simpleCollect\';

dateString = datestr(now,'yyyymmddTHHMMSS');

newDataName = sprintf('syntData_%s',dateString);

if ~exist(fullfile(gtDir, newDataName), 'dir')
    mkdir(fullfile(gtDir, newDataName));
end

segmentationList = dir(fullfile(segmentationDir,['*' segmentationExtension]));

for segmInd = 1:length(segmentationList)
    
    
    segmName = segmentationList(segmInd).name;
    
    [~,segmImageId,~] = fileparts(segmName);
    
    
%% read a 'good' segmentation
    
    segmentation = imread(fullfile(segmentationDir, segmName));
%     figure; imagesc(segmentation); axis image; title(sprintf('init segmentation for image %s',segmImageId));

    % clear border objects to discard a family of false shapes of half
    % objects
    segmentation = clearBorderObjects(segmentation);
%     figure; imagesc(segmentation); axis image; title(sprintf('init segmentation for image %s',segmImageId));
    

%% collect features from cells

    props = regionprops(segmentation, 'Area','Solidity', 'Eccentricity', 'MajorAxisLength', 'MinorAxisLength', 'Perimeter');
        
%% read raw image for colour statistics

    rawImage = imread(fullfile(rawImageDir, segmName));
    bgColor = zeros(1,3);
    for chInd = 1:3
        ch = rawImage(:,:,chInd);
        bgColor(chInd) = double(median(ch(segmentation==0)))/256.0;
    end
    
    colourProps1 = regionprops(segmentation, rawImage(:,:,1), 'MeanIntensity', 'MaxIntensity', 'MinIntensity');
    colourProps2 = regionprops(segmentation, rawImage(:,:,2), 'MeanIntensity', 'MaxIntensity', 'MinIntensity');
    colourProps3 = regionprops(segmentation, rawImage(:,:,3), 'MeanIntensity', 'MaxIntensity', 'MinIntensity');
    colourProps1(cat(1,props.Area)<1) = [];
    colourProps2(cat(1,props.Area)<1) = [];
    colourProps3(cat(1,props.Area)<1) = [];

    fgColors = [cat(1,colourProps1.MeanIntensity) cat(1,colourProps3.MeanIntensity) cat(1,colourProps3.MeanIntensity)]/256.0;
    
    %TODO connect here to database
    props(cat(1,props.Area)<1) = [];
    
    representativeFeatureVectors = zeros(length(props), 5); % area, eelongation, circularity, eccentricity, solidity
    representativeFeatureVectors(:,1) = cat(1, props.Area);
    representativeFeatureVectors(:,2) = cat(1, props.MajorAxisLength)./cat(1,props.MinorAxisLength);
    representativeFeatureVectors(:,3) = cat(1, props.Area)./cat(1,props.Perimeter).^2;
    representativeFeatureVectors(:,4) = cat(1, props.Eccentricity);
    representativeFeatureVectors(:,5) = cat(1, props.Solidity);
    
%% run simcep_dsb_options + finetune numbers

    simcep_dsb_options;
    
    population.template = ones(size(segmentation));
    
    imageArea = numel(population.template);
    imageCoveredArea = sum(cat(1,props.Area));
    
    if imageCoveredArea/imageArea > coverThreshold;
        population.N = length(props);
    else
        population.N = int32(imageArea*coverThreshold / mean(cat(1,props.Area)));
    end
    
    cell_obj.nucleus.representativeFeatureVectors = representativeFeatureVectors;
    cell_obj.nucleus.radius = sqrt(mean(cell_obj.nucleus.representativeFeatureVectors(:,1))/pi);

%% run simcep_dsb and save images and masks

    for imInd = 1:numOfOutputImages
        
        synthImageName = sprintf('%s_%s_%03d.png', segmImageId, dateString, imInd);
        fprintf('Generating synthetic image %s\n', synthImageName);
        [~,synthImageId,~] = fileparts(synthImageName);
        
        [RGB, BW, features, cellStructs] = simcep_dsb(population, cell_obj, measurement); 
%         figure; imagesc(BW(:,:,3)); axis image; title(sprintf('%s_%02d',segmImageId,imInd),'Interpreter','none');

        nucleiMergedMasks = uint16(squeeze(BW(:,:,3)));
        %save merged labeled mask
        imwrite(nucleiMergedMasks,fullfile(gtDir,newDataName,synthImageName), 'Bitdepth', 16);
        
        %save individual masks
        masksDir = fullfile(gtDir,newDataName,synthImageId,'masks');
        if ~exist(masksDir,'dir')
            mkdir(masksDir);
        end
        uniqueValues = unique(nucleiMergedMasks);
        if uniqueValues(1) == 0
            uniqueValues(1) = [];
        end
        for maskInd = 1:length(uniqueValues)
            currentMask = uint8(nucleiMergedMasks==uniqueValues(maskInd))*255;
            imwrite(currentMask, fullfile(masksDir, sprintf('image_%03d.png',maskInd)));
        end
        
        % creates a customized colormap for the generated mask image based
        % on the segmented cell intensities
        
        randNums = randi(length(colourProps1), length(uniqueValues), 1);
        colorMapForFgObjects = fgColors(randNums,:);
        rgbMaskImage = label2rgb(nucleiMergedMasks, colorMapForFgObjects, bgColor);
        
        colourMasksDir = fullfile(gtDir,newDataName,synthImageId,'colourmasks');
        if ~exist(colourMasksDir,'dir')
            mkdir(colourMasksDir);
        end
        % saving coloured mask images
        imwrite(rgbMaskImage,fullfile(colourMasksDir,synthImageName));
    end

end