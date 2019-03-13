function probMapMRCNNIntersect(mRCNNInput, probMapsInput, masksOutDir, intRatioTresh, verbose)
    
    if nargin < 5
        verbose = false;
    end
    % Dependency: generateDataFromDSBInput

    % This script takes the M-RCNN predictions in kaggle format and the probability
    % maps, and then filters out such predicted object (the single object mask
    % file) where the intersections are below a predefined threshold.

    % The M-RCNN prediction directory
    %mRCNNInput='/home/biomag/etasnadi/pred-result_model515';

    % The probability maps dir
    %probMapsInput = '/home/biomag/etasnadi/probmaps-splited-kaggle-format-sep/tissue-gathered';

    % The result dir, where the objects are put on a binary image for the style
    % transfer
    %intersectedMasksOutDir = strcat('/home/biomag/etasnadi/rcnn-probmaps-intersect-binary-masks', '_', num2str(intRatioTresh));

    intersectedMasksOutDir = fullfile(masksOutDir, num2str(intRatioTresh));
    disp(['Intersected masks destination: ', intersectedMasksOutDir]);
    mkdir(intersectedMasksOutDir);
    
    
    masksOutDirIntEnc = fullfile(masksOutDir, ['int-enc-', num2str(intRatioTresh)]);
    
    elems = dir(mRCNNInput);
    nElems = size(elems, 1);
    for id = 1:nElems
        elem = elems(id);
        if elem.isdir && ~strcmp(elem.name, '.') && ~strcmp(elem.name, '..')
            trainingsample = elem.name;
            sampleMasksPath = fullfile(mRCNNInput, trainingsample, 'masks');
            disp(['Processing: ', trainingsample]);

            probmap = imread(fullfile(probMapsInput, strcat(trainingsample, '.png')));

            masksDirContent = dir(strcat(sampleMasksPath, '/*.png'));
            nMaskObjs = size(masksDirContent, 1);
            sampleMasksOutPath=fullfile(intersectedMasksOutDir, trainingsample, 'masks');
            mkdir(sampleMasksOutPath);
            for maskId = 1:nMaskObjs
                objectMaskFname = masksDirContent(maskId).name;
                objectMaskPath = fullfile(sampleMasksPath, objectMaskFname);
                objectMask = imread(objectMaskPath);
                objectMaskOutPath=fullfile(sampleMasksOutPath, objectMaskFname);
                if(cmppred(objectMask, probmap, intRatioTresh))
                    if verbose
                        disp(strcat('Copying:', objectMaskPath, '->', objectMaskOutPath))
                    end
                    copyfile(objectMaskPath, objectMaskOutPath);
                end
            end
        end
    end
    
    % Merge the mask into an intensity encoded image.
    generateDataFromDSBInput(intersectedMasksOutDir, 'outputDir', masksOutDirIntEnc, 'outputDataType', 'mergedMaskLabelledImage');
end