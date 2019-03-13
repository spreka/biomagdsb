function features=  buildFeatureDatabase(rawImageDir, segmDir)
%reads all objects and extracts all features of objects
% features: [area eccentricity mean(ch1Obj) mean(ch2Obj) mean(ch3Obj) cov(ch1Obj, ch2Obj, ch3Obj) 
%            mean(ch1) mean(ch2) mean(ch3) cov(ch1, ch2, ch3) mean(ch1BG) mean(ch2BG) mean(ch3BG) cov(ch1BG, ch2BG, ch3BG)] 

% TODO? check features

imagesList = dir(fullfile(rawImageDir,'*.png'));

features = containers.Map();

for i=1:length(imagesList)
    
    [~,imageBaseName,~] = fileparts(imagesList(i).name);
    rawImage = im2double(imread(fullfile(rawImageDir,[imageBaseName '.png'])));
    segmentation = imread(fullfile(segmDir,[imageBaseName '.tiff']));
    segmentation = relabelImage(segmentation);
    commonProps = regionprops(segmentation, 'Area', 'Eccentricity');
    
    
    % shape features
    currentFeatures = zeros(length(commonProps),12);
    currentFeatures(:,1) = cat(1,commonProps.Area);
    currentFeatures(:,2) = cat(1,commonProps.Eccentricity);
    
%     currentFeatures{length(commonProps)} = [];
    pIntensities = cell(3,1);
    
    for chInd = 1:3
        channel = rawImage(:,:,chInd);
        props = regionprops(segmentation, channel, 'PixelValues');
        pIntensities{chInd,1} = {props.PixelValues};
    end
        
    currentFeatures(:,3) = [cellfun( @(x) mean(x), pIntensities{1} )];
    currentFeatures(:,4) = [cellfun( @(x) mean(x), pIntensities{2} )];
    currentFeatures(:,5) = [cellfun( @(x) mean(x), pIntensities{3} )];
    
    for objInd=1:length(props)
        objIntensities = [pIntensities{1}{objInd} pIntensities{2}{objInd} pIntensities{3}{objInd}];
        objCov = cov(objIntensities);
        currentFeatures(objInd,6:6+9-1) = reshape(objCov, [], numel(objCov));
    end
    
    pInt = [cat(1,  pIntensities{1,1}{:}),...
            cat(1,  pIntensities{2,1}{:}),...
            cat(1,  pIntensities{3,1}{:})];
    
    fgMeans = mean(pInt,1);
    fgCov = cov(pInt);
    fgFeatures = [reshape(fgMeans, [], numel(fgMeans)) reshape(fgCov, [], numel(fgCov))];
    currentFeatures(:,15:15+12-1) = repmat(fgFeatures,length(props),1);
    
    pIntensities = cell(3,1);
    for chInd = 1:3
        channel = rawImage(:,:,chInd);
        bgProps = regionprops(segmentation==0,channel, 'PixelValues');
        pIntensities{chInd,1} = {bgProps.PixelValues};
    end
    
    pInt = [cat(1,  pIntensities{1,1}{:}),...
            cat(1,  pIntensities{2,1}{:}),...
            cat(1,  pIntensities{3,1}{:})];
    
    bgMeans = mean(pInt,1);
    bgCov = cov(pInt);
    bgFeatures = [reshape(bgMeans, [], numel(bgMeans)) reshape(bgCov, [], numel(bgCov))];
    currentFeatures(:,15+12:15+12+12-1) = repmat(bgFeatures,length(props),1);
    
    features(imageBaseName) = currentFeatures;
    
end