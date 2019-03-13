function postProcCodeFINAL(inMainFolder,outMainFolder,...
    scaleFolder1,scaleFolder2,gtFolder,sizeSelectedFolder,probFolder,...
    finalFolder,params)
%% ---------- input parameters ---------- %%
%% postprocess test
if nargin==0 % use default parameters; CAUTION: this will NOT work bacause of the paths!
inMainFolder = '/ssd/spreka/mrcnn_output/postprocessing/largeAug_0618_conf06_val_NEWNEW/in/';
outMainFolder = '/ssd/spreka/mrcnn_output/postprocessing/largeAug_0618_conf06_val_NEWNEW/out/';
scaleFolder1='V_C2x60/'; %'1x';
scaleFolder2='V_C4x60/';

gtFolder = '/ssd/spreka/mrcnn_input/val/masks/';
sizeSelectedFolder = 'master';
probFolder = 'probmaps/ensembled/';
finalFolder = 'masterBlaster';

% segmentation parameters:
%scaleThrsh, probThresh, erosionRadius, dilationRadius, minSize, minOverlap, maxV, carea, median size
params=[76, 32, 0, 1, 44, 0017, 32, 25, 21]; % optimal parameters

end

scaleTh=params(1)/100;
probTh=params(2);
er=params(3);
dil=params(4);
minSize=params(5);
minOverlapp=params(6);
maxVv=params(7);
cAreav=params(8);
medianv=params(9);

%% ---------- pre-process outputs ---------- %%

if ~exist(fullfile(outMainFolder, scaleFolder1),'dir') && ~exist(fullfile(outMainFolder, scaleFolder2),'dir')
    postProcessAllDataFINAL([fullfile(inMainFolder, scaleFolder2) filesep], [fullfile(outMainFolder, scaleFolder2) filesep], minSize);
    postProcessAllDataFINAL([fullfile(inMainFolder, scaleFolder1) filesep], [fullfile(outMainFolder, scaleFolder1) filesep], minSize);
end

%% ---------- run post-processing ---------- %
% ***************************************** %
resultFolder=outMainFolder;


%parameters
removeLastUnderScore = 1;
minOverlap = 0.0;

params = [0];

for o1=1:numel(params)
    
    param = params(o1);
    
    mkdir([resultFolder sizeSelectedFolder]);
    mkdir([resultFolder finalFolder]);
    
    outFolderName1 = [resultFolder scaleFolder1 filesep];
    outFolderName2 = [resultFolder scaleFolder2 filesep];
    
    fileList = dir(fullfile(gtFolder,'*.tiff'));
    
    for i=1:numel(fileList)
        
        %% check file existance
        if ~exist([resultFolder scaleFolder1  filesep fileList(i).name], 'file')
            in = imread([gtFolder fileList(i).name]);
            imwrite(uint16(in * 0), [resultFolder scaleFolder1  filesep fileList(i).name])
        end
        if ~exist([resultFolder scaleFolder2  filesep fileList(i).name], 'file')
            in = imread([gtFolder fileList(i).name]);
            imwrite(uint16(in * 0), [resultFolder scaleFolder2  filesep fileList(i).name])
        end
        
        inImg = imread([resultFolder scaleFolder1  filesep fileList(i).name]);
        [median_size, std_size] = estimateCellSizeFromMask(inImg);
        if isnan(median_size)
            inImg = imread([resultFolder scaleFolder2  filesep fileList(i).name]);
            [median_size, std_size] = estimateCellSizeFromMask(inImg);
        end

        %disp([fileList(i).name ': ' num2str(median_size)]);
       
        if median_size > medianv
            inImg = imread([outFolderName1 fileList(i).name]);
        else
            inImg1 = imread([outFolderName1 fileList(i).name]);
            inImg2 = imread([outFolderName2 fileList(i).name]);
            inImg = mergeTwoFilesFINAL(inImg1, inImg2, scaleTh);
        end
 
        imwrite(inImg, [resultFolder sizeSelectedFolder filesep fileList(i).name]);
        fprintf('#'); if mod(i, 60)==0, fprintf('\n'); end
    end
    fprintf('\n');
    
    
    imageList = dir(fullfile(resultFolder,sizeSelectedFolder,'*.tiff'));
    
    for i=1:numel(imageList)
        
        mask = imread([resultFolder sizeSelectedFolder filesep imageList(i).name]);
        [mx, my] = size(mask);
        probMap = (imread([probFolder filesep imageList(i).name(1:end-5) '.png']));
        [px, py] = size(probMap);
        if px == mx  && my == py
        else
            probMap = imresize(probMap, [mx, my]);
        end
        
        
        probThres = probTh;
        scaleUNET = 32;
        probMapReal = double(probMap)/65535;
        % param1
        probMap = probMap >= probThres* (65535/scaleUNET);
        
        maskBig = double(dilateLabelledMasks(mask, dil)); %1));
        maskSmall = double(erodeLabelledMasks(mask, er)); %1));
        
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
                if maxv > maxVv
                    outPos = find(labelledBlank == maxi);

                    outFinal(outPos) = index;
                    index = index + 1;
                end
            end
        end
        
        
        
        % Revert too large objects
        
        %% remove false positives
        index = 1;
        outNoFPos = out * 0;
        % param 5
        minOverlap = minOverlapp/10000; %0009/10000;%0.65;

        
        
        for j=1:max(outFinal(:))
            pix = find(outFinal == j);
            cArea = numel(pix);
            probMapSum = sum(probMap(pix));
            if probMapSum / cArea >= minOverlap
                if cArea>=cAreav
                    outNoFPos(pix) = index;
                    index = index + 1;
                end
            else
                if cArea >= cAreav*4
                    outNoFPos(pix) = index;
                    index = index + 1;
                else
                    fprintf('%i %f\n', cArea, probMapSum / cArea);
                    disp([imageList(i).name ', Removed: ' num2str(j)]);
                end
            end
        end
        
        % Add false negatives based on UNET and

        imwrite(uint16(outNoFPos), [resultFolder finalFolder filesep imageList(i).name]);
        
    end
    
end