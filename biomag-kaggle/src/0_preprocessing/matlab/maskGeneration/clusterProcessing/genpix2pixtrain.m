function genpix2pixtrain(predClustersDir, inputClustersDir, styTransferTrainDir)
    disp(['mkdir' styTransferTrainDir]);
    mkdir(styTransferTrainDir);
    
    predConts = dir(predClustersDir);
    for clusId=1:numel(predConts)
        clusName = predConts(clusId).name;
        if ~strcmp(clusName, '.') && ~strcmp(clusName, '..')
            actPredClusDir = fullfile(predClustersDir, clusName);
            disp(['Cluster name: ' clusName]);
            actStyTransferClusDir = fullfile(styTransferTrainDir, clusName, 'train');
            
            disp(['mkdir' actStyTransferClusDir]);
            mkdir(actStyTransferClusDir);
            
            predMasks = dir(fullfile(actPredClusDir)); % '*.tiff' is removed there to be more general
            for predMaskId=3:numel(predMasks) % go from 3 to avoid . and ..
                predMaskName = predMasks(predMaskId).name;
                [~, predMaskNameWOExt, ~] = fileparts(predMaskName);
                rawImagePath = fullfile(inputClustersDir, clusName, [predMaskNameWOExt '.png']);
                predMaskPath = fullfile(actPredClusDir, predMaskName);
                styTransferInputPath = fullfile(actStyTransferClusDir, [predMaskNameWOExt '.png']);
                disp(['Writing to: ' styTransferInputPath ' (' rawImagePath '+' predMaskPath ')']);
                if strcmp(clusName,'group_020')
                    disp('problem');
                end
                createMask(rawImagePath, predMaskPath, styTransferInputPath);
            end
        end
    end
end

function createMask(rawImagePath, maskPath, styTransferTrainImagePath)
    mask = imread(maskPath);
    [rawImg,maps] = imread(rawImagePath);    
    if size(rawImg,3) ~= 3
        if size(maps,2)==1 || isempty(maps)
            rawImg = repmat(rawImg,1,1,3);            
        else
            rawImg = ind2rgb(rawImg,maps);                
        end
    end

    maskPart = zeros(size(mask));
    for ch = 1:3
        maskPart(:,:,ch)=mask(:,:);
    end
    targetIm = [rawImg, (maskPart>0)*255];
    imwrite(targetIm, styTransferTrainImagePath);
end