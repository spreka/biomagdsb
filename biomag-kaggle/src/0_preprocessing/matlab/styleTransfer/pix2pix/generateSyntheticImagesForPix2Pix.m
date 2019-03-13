% Script to generate train style images and synthetic nuclei images for
% pix2pix framework.
% Required directories:
%       - directory with the raw images to generate train data.
%       - directory with an initial segmentation: they are used for both
%         train data and synthetic data generation.
%
% The script generates the train data inside 'maskedstyles' folder, it's
% content should be copied to the 'style/train' folder.
% The synthetic masks will be saved to the 'syntheticforpix2pix' folder,
% it's content belongs to the 'masks/test' folder..

% Number of synthetic images per input image
numOfOutputImages = 10;

% Approximate covering of the output image
coverThreshold = 0.3;

% Probability for assiging simulated cell into a cluster. Otherwise the
% cell is uniformly distributed on the image.
clustProb = 0.8;
spatvar = 0.2;

% Directory where the initial segmentation is stored
segmentationDir = '/home/biomag/etasnadi/augmented-masks/o';
segmentationExtension = '.png';

% Directory where to save generated synthetic images
gtDir = '/home/biomag/etasnadi/augmented-masks/g';

% Directory with the input raw images (required for style generation)
rawImageDir = '/home/biomag/etasnadi/augmented-masks/i';

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
    
    rawImage = imread(fullfile(rawImageDir, segmName));
    
    % creates a concantenated image for styles
        
    colourMasksDir = fullfile(gtDir,newDataName,'maskedstyles');
    if ~exist(colourMasksDir,'dir')
        mkdir(colourMasksDir);
    end
    % saving coloured mask images
    imwrite(cat(2,rawImage, repmat(uint8(segmentation>0)*255,1,1,3)),fullfile(colourMasksDir,segmName));
    
    pix2pixDir = fullfile(gtDir,newDataName,'syntheticforpix2pix');
    if ~exist(pix2pixDir,'dir')
        mkdir(pix2pixDir);
    end
    
%     figure; imagesc(segmentation); axis image; title(sprintf('init segmentation for image %s',segmImageId));

    % clear border objects to discard a family of false shapes of half
    % objects
    segmentation = clearBorderObjects(segmentation);
%     figure; imagesc(segmentation); axis image; title(sprintf('init segmentation for image %s',segmImageId));
    

%% collect features from cells

    props = regionprops(segmentation, 'Area','Solidity', 'Eccentricity', 'MajorAxisLength', 'MinorAxisLength', 'Perimeter');
        
%% read raw image for colour statistics

    
%     bgColor = zeros(1,3);
%     for chInd = 1:3
%         ch = rawImage(:,:,chInd);
%         bgColor(chInd) = double(median(ch(segmentation==0)))/256.0;
%     end
%     
%     colourProps1 = regionprops(segmentation, rawImage(:,:,1), 'MeanIntensity', 'MaxIntensity', 'MinIntensity');
%     colourProps2 = regionprops(segmentation, rawImage(:,:,2), 'MeanIntensity', 'MaxIntensity', 'MinIntensity');
%     colourProps3 = regionprops(segmentation, rawImage(:,:,3), 'MeanIntensity', 'MaxIntensity', 'MinIntensity');
%     colourProps1(cat(1,props.Area)<1) = [];
%     colourProps2(cat(1,props.Area)<1) = [];
%     colourProps3(cat(1,props.Area)<1) = [];
% 
%     fgColors = [cat(1,colourProps1.MeanIntensity) cat(1,colourProps3.MeanIntensity) cat(1,colourProps3.MeanIntensity)]/256.0;
    
    %TODO connect here to database
    props(cat(1,props.Area)<1) = [];
    
    representativeFeatureVectors = zeros(length(props), 5); % area, eelongation, circularity, eccentricity, solidity
    representativeFeatureVectors(:,1) = cat(1, props.Area);
    representativeFeatureVectors(:,2) = cat(1, props.MajorAxisLength)./cat(1,props.MinorAxisLength);
    representativeFeatureVectors(:,3) = cat(1, props.Area)./cat(1,props.Perimeter).^2;
    representativeFeatureVectors(:,4) = cat(1, props.Eccentricity);
    representativeFeatureVectors(:,5) = cat(1, props.Solidity);
    
    % TODO discard outliers
%% run simcep_dsb_options + finetune numbers

    simcep_dsb_options;
    
%     population.template = ones( size(segmentation) );
    population.template = ones(256, 256);
    population.clustprob = clustProb;
    
    population.spatvar = spatvar;
    
    imageArea = numel(population.template);
    imageCoveredArea = sum(cat(1,props.Area));
    
    if imageCoveredArea/imageArea > coverThreshold;
        population.N = length(props);
    else
        population.N = int32( imageArea*coverThreshold / mean(cat(1,props.Area)) );
    end
    
    cell_obj.nucleus.representativeFeatureVectors = representativeFeatureVectors;
    cell_obj.nucleus.radius = sqrt(mean(cell_obj.nucleus.representativeFeatureVectors(:,1))/pi);

%% run simcep_dsb and save images and masks

    for imInd = 1:numOfOutputImages
        
        synthImageName = sprintf('%s_%s_%03d.png', segmImageId, dateString, imInd);
        fprintf('Generating synthetic image %s\n', synthImageName);
        [~,synthImageId,~] = fileparts(synthImageName);
        
        tic;
        [RGB, BW, features, cellStructs] = simcep_dsb(population, cell_obj, measurement); 
%         figure; imagesc(BW(:,:,3)); axis image; title(sprintf('%s_%02d',segmImageId,imInd),'Interpreter','none');

        generationTime = toc;
        fprintf('Ellapsed time for image generation: %0.2f\n',generationTime);
        
        nucleiMergedMasks = uint16(squeeze(BW(:,:,3)));
        %save merged labeled mask
        imwrite(nucleiMergedMasks, fullfile(gtDir,newDataName,synthImageName), 'Bitdepth', 16);
        
        %save pix2pix image
        imwrite( repmat( uint8(nucleiMergedMasks>0)*255, 1,2,3), fullfile(pix2pixDir, synthImageName) ); 
        
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
        
        
    end

end