clear all;
%load('testSet.mat');
% inputs
resultFolder = 'c:\projects_tmp\NucleiCompetition\results\20180415_optimizer\';
gtFolder = 'c:\projects_tmp\NucleiCompetition\results\20180415_optimizer\validation\validation\masks\';
scaleFolder1 = '1x'; %'1x';
scaleFolder2 = '2x';
sizeSelectedFolder = 'master';
probFolder = 'probmaps\ensembled\';
finalFolder = 'masterBlaster';


%parameters
removeLastUnderScore = 1;
minOverlap = 0.0;

mkdir([resultFolder sizeSelectedFolder]);
mkdir([resultFolder finalFolder]);

outFolderName1 = [resultFolder scaleFolder1 '\'];
outFolderName2 = [resultFolder scaleFolder2 '\'];



%fileList = dir([resultFolder scaleFolder1 '\*.tiff']);
fileList = dir([gtFolder '\*.tiff']);

for i=1:numel(fileList)
    
    %% check file existance
    if ~exist([resultFolder scaleFolder1  '/' fileList(i).name], 'file')
        in = imread([gtFolder fileList(i).name]);
        imwrite(uint16(in * 0), [resultFolder scaleFolder1  '/' fileList(i).name])
    end
    if ~exist([resultFolder scaleFolder2  '/' fileList(i).name], 'file')
        in = imread([gtFolder fileList(i).name]);
        imwrite(uint16(in * 0), [resultFolder scaleFolder2  '/' fileList(i).name])
    end
    
    
    %%%%%%%%%
    %%%%%% CHANGE1
    inImg = imread([resultFolder scaleFolder1  '/' fileList(i).name]);
    [median_size, std_size] = estimateCellSizeFromMask(inImg);
    if isnan(median_size)
        inImg = imread([resultFolder scaleFolder2  '/' fileList(i).name]);
        [median_size, std_size] = estimateCellSizeFromMask(inImg);
    end
    %%%%%%%%%
    %%%%%% CHANGE1 END

    
    disp([fileList(i).name ': ' num2str(median_size)]);

    %%%%%%%%%
    %%%%%% CHANGE2
    if median_size > 20
        inImg = imread([outFolderName1 fileList(i).name]);
    else
        inImg1 = imread([outFolderName1 fileList(i).name]);
        inImg2 = imread([outFolderName2 fileList(i).name]);
        inImg = mergeTwoFiles(inImg1, inImg2);
    end
    %%%%%%%%%
    %%%%%% CHANGE2 END

    
    
    imwrite(inImg, [resultFolder sizeSelectedFolder '\' fileList(i).name]);
    
end



imageList = dir([resultFolder sizeSelectedFolder  '\*.tiff']);

for i=1:numel(imageList)
    
    mask = imread([resultFolder sizeSelectedFolder '\' imageList(i).name]);
    [mx, my] = size(mask);
    probMap = (imread([resultFolder probFolder '\' imageList(i).name(1:end-5) '.png']));
    [px, py] = size(probMap);
    if px == mx  && my == py
    else
        probMap = imresize(probMap, [mx, my]);
    end
    
    
    %%%%%%%%%
    %%%%%% CHANGE3    
    %%% OK %%%
    probThres = 18;
    scaleUNET = 32;
    probMapReal = double(probMap)/65535;
    % param1
    probMap = probMap >= probThres * (65535 / scaleUNET);
    %%%%%%%%%
    %%%%%% CHANGE3 END

    
    
    %mask = double(erodeLabelledMasks(mask, 1));
    % param 2, 3
    maskBig = double(dilateLabelledMasks(mask, 1));
    maskSmall = double(erodeLabelledMasks(mask, 1));
    
    ring = double(maskBig - maskSmall);
    ring = ring .* probMap;
    
    out = ring + maskSmall;
    
    outFinal = out * 0;
    
    index = 1;
    for j=1:max(out(:))
        
        blank = out * 0;
        pos = find(out == j);
        blank(pos) = 1;
        labelledBlank = bwlabel(blank, 4);
        stats = regionprops(blank, 'Area');
        if ~isempty(stats)
            [maxv, maxi] = max(stats.Area);
            % param 4
            if maxv > 30
                %                 if maxv > 630
                %                     outPos = find(mask == j);
                %                 else
                outPos = find(labelledBlank == maxi);
                %                 end
                outFinal(outPos) = index;
                index = index + 1;
            end
        end
    end
    
    
    
    % Revert too large objects
    
    %    outFinal = mask;
    
    %% remove false positives
    index = 1;
    outNoFPos = out * 0;
    % param 5
    
    %%%%%%%%%
    %%%%%% CHANGE4
    minOverlap = 0;%0.65;
    %%%%%%%%%
    %%%%%% CHANGE4 END

    
    for j=1:max(outFinal(:))
        pix = find(outFinal == j);
        cArea = numel(pix);
        probMapSum = sum(probMap(pix));
        if probMapSum / cArea >= minOverlap
            if cArea>=21
                outNoFPos(pix) = index;
                index = index + 1;
            end
        else
            disp([imageList(i).name ', Removed: ' num2str(j)]);
        end
    end
    
    % Add false negatives based on UNET and
    
    %% 4 conn?
    imwrite(uint16(outNoFPos), [resultFolder finalFolder '\' imageList(i).name]);
    
end

% test eval
scores = evaluation(gtFolder, [resultFolder finalFolder '\'], '', 'eval.csv');
% 
% runLengthEncodeFolder([resultFolder finalFolder]);
