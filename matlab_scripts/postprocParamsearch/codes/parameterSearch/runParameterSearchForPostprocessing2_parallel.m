%% postprocess script
% clear;

gtFolder = '/media/baran/LinuxData/Downloads/Challange/Optimizer/validationData_20180415/validation/masks/'
inFolder = '/media/baran/LinuxData/Downloads/Challange/Optimizer/predicetedData_20180415/'
inProbmapFolders = {'/media/baran/LinuxData/Downloads/Challange/Optimizer/probmapData_20180415/ensembled'};
outFolder = '/media/baran/LinuxData/Downloads/Challange/Optimizer/kaggle_data2_21080415/results/';
resultsFile = '/media/baran/LinuxData/Downloads/Challange/Optimizer/kaggle_data2_21080415/results/paramsearchresult.txt';

%gtFolder = '/media/baran/LinuxData/Downloads/Challange/Optimizer/kaggle_data_21080411/groundtruth2/'
%inFolder = '/media/baran/LinuxData/Downloads/Challange/Optimizer/kaggle_data_21080411/'
%inProbmapFolders = {'/media/baran/LinuxData/Downloads/Challange/Optimizer/kaggle_data_21080411/probmaps/ensembled/'};
%outFolder = '/media/baran/LinuxData/Downloads/Challange/Optimizer/kaggle_data_21080411/results/';

mkdir(outFolder);

fid = fopen(resultsFile, 'w');

smallScaleImagesFolder.scale = 1;
smallScaleImagesFolder.name = fullfile(inFolder, 'V_C2x50', '');

bigScaleImagesFolder.scale = 2;
bigScaleImagesFolder.name = fullfile(inFolder, 'V_C4x50', '');


%% settings
conn = 8;

%{
% TODO test param
erosionRadii = [0:3]
dilationRadii = [0:3]
minSizes = [5:10:75]
scaleThreshs = [10:2:30];
probThresholds = [8656-500:100:8656+500];
% resize working image to original size
resizeFactor = 0.5;

parameterArray(numel(erosionRadii)*numel(dilationRadii)*numel(minSizes)*numel(scaleThreshs)*numel(probThresholds)) = struct('id',0, 'parameters', struct('minSize',0, 'scaleThresh',0,'probThresh',0,'erosionRadius',0,'dilationRadius',0));

ind = 0;
for minSizeInd = 1:length(minSizes)
    for scaleThrshInd = 1:length(scaleThreshs)
        for probThreshInd = 1:length(probThresholds)
            for erosionRadiusInd = 1:length(erosionRadii)
                for dilationRadiusInd = 1:length(dilationRadii)
                    ind = ind+1;
                    parameterArray(ind) = struct('id',ind, 'parameters', struct('minSize',minSizes(minSizeInd), 'scaleThresh',scaleThreshs(scaleThrshInd),'probThresh',probThresholds(probThreshInd),'erosionRadius',erosionRadii(erosionRadiusInd),'dilationRadius',dilationRadii(dilationRadiusInd)));
                end
            end
        end
    end
end

resStringCellArray = cell(numel(parameterArray),1);
%}

saveMultiScaleImages = 0;
saveAllParamsResults = 0;
evalAllResults = 1;

p = gcp('nocreate');
if isempty(p)
    parpool('local',feature('numcores'));
end
% delete(gcp('nocreate'));


%% making folder structure

if saveMultiScaleImages
    masterScaleFolder = fullfile(outFolder, 'masterScale');
    mkdir(masterScaleFolder);
end

% TODO change it for all parameters
if saveAllParamsResults
    for scaleThrsh = scaleThreshs
        for erosionRadius = erosionRadii
            for dilationRadius = dilationRadii
                for minSize = minSizes
                    outFolderCaseName = sprintf('masterScale_e%d_d%d_min%d', erosionRadius, dilationRadius, minSize);
                    outFolderCaseFullPath = fullfile(outFolder,outFolderCaseName);
                    mkdir(outFolderCaseFullPath);
                end
            end
        end
    end
end

sumProbMap = readSegmentation(inProbmapFolders{1}, '.png');

gtMap = readSegmentation(gtFolder, '.tiff');
allKeys = gtMap.keys();

smallScaleImagesMap = readSegmentation(smallScaleImagesFolder.name,'.tiff');
bigScaleImagesMap = readSegmentation(bigScaleImagesFolder.name,'.tiff');

tic

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
for ik=1:length(allKeys)
                     
    if (smallScaleImagesMap.isKey(allKeys{ik}) == false)        
        fprintf('missing small scale image: %s\n', allKeys{ik});
        [imX, imY, imZ] = size(gtMap(allKeys{ik}));
        smallScaleImagesMap(allKeys{ik}) = uint16(ones(round(imX / 2), round(imY / 2), 1));
        imwrite(smallScaleImagesMap(allKeys{ik}), strcat(fullfile(smallScaleImagesFolder.name, allKeys{ik}), '.tiff'));
    end
    
    if (bigScaleImagesMap.isKey(allKeys{ik}) == false)
        [imX, imY, imZ] = size(gtMap(allKeys{ik}));
        fprintf('missing big scale image: %s\n', allKeys{ik});                
        bigScaleImagesMap(allKeys{ik}) = uint16(ones(imX * 2, imY * 2, 1));                       
        imwrite(bigScaleImagesMap(allKeys{ik}), strcat(fullfile(bigScaleImagesFolder.name, allKeys{ik}), '.tiff'));
    end
    
end

tic

%postprocess all segmentation files: fill holes and merge touching ones
%%{
for ik=1:length(allKeys)
    smallScaleImagesMap(allKeys{ik}) = removeObjectWithinObject(smallScaleImagesMap(allKeys{ik}));
    smallScaleImagesMap(allKeys{ik}) = mergeTouchingObjects(smallScaleImagesMap(allKeys{ik}), conn);    
    
    bigScaleImagesMap(allKeys{ik}) = removeObjectWithinObject(bigScaleImagesMap(allKeys{ik}));
    bigScaleImagesMap(allKeys{ik}) = mergeTouchingObjects(bigScaleImagesMap(allKeys{ik}), conn);
end
%writeSegmentation(outFinalImageMap, '/media/baran/LinuxData/Downloads/Challange/Optimizer/temp_outFinal', '.tiff');
%%}

%{
% main loop
parfor i=1:length(parameterArray)
    parameters = parameterArray(i).parameters;
    postProcessedMap = postProcessSegmentation2( struct('map', smallScaleImagesMap, 'scale', smallScaleImagesFolder.scale), struct('map', bigScaleImagesMap, 'scale', bigScaleImagesFolder.scale), sumProbMap, parameters);
    if evalAllResults
        score = evaluation2(gtMap, postProcessedMap);
%         fprintf('Score for corrected segmentations (scaleTh:%d, er:%dpx, dil:%dpx, min:%d, probTh:%d): %0.3f\n',...
%                 scaleThresh,erosionRadius,dilationRadius,minSize,probThresh,mean(cell2mat(score.values)));
%         fidCurrent = fopen(fullfile(outFolder,sprintf('%06d_score.txt',parameterArray(i).id)),'w'); 
%         fprintf(fidCurrent,'Score for corrected segmentations (scaleTh:%d, er:%dpx, dil:%dpx, min:%d, probTh:%d): %0.3f\n',...
%                 parameters.scaleThresh,parameters.erosionRadius,parameters.dilationRadius,parameters.minSize,parameters.probThresh,mean(cell2mat(score.values)));
%         fclose(fidCurrent);
        resStringCellArray{i} = sprintf('Score for corrected segmentations (scaleTh:%d, er:%dpx, dil:%dpx, min:%d, probTh:%d): %0.3f\n',...
                         parameters.scaleThresh,parameters.erosionRadius,parameters.dilationRadius,parameters.minSize,parameters.probThresh,mean(cell2mat(score.values)));
    end
end

for i=1:length(resStringCellArray)
    fprintf(fid,[resStringCellArray{i}]);
end
%}

toc

%{

for minSize = minSizes
    % discard small objects 1st round
    for i=1:length(allKeys)
        smallScaleImage = removeSmallObjects(smallScaleImagesMap(allKeys{i}), minSize);
        smallScaleImagesMap(allKeys{i}) = imresize(smallScaleImage, size(sumProbMap(allKeys{i})), 'nearest');
        bigScaleImage = removeSmallObjects(bigScaleImagesMap(allKeys{i}), minSize);
        bigScaleImagesMap(allKeys{i}) = imresize(bigScaleImage, size(sumProbMap(allKeys{i})), 'nearest');
    end
    for scaleThrsh = scaleThreshs
        % merging images from 2 folder
        mergedImgMap = mergeScalesMap2(struct('map',smallScaleImagesMap, 'scale', 1), struct('map',bigScaleImagesMap, 'scale', 2), scaleThrsh);
        for probThresh = probThresholds
            for erosionRadius = erosionRadii
                for dilationRadius = dilationRadii
                    
                    %                     fprintf('Calc corrected segmentations (scaleTh:%d, er:%dpx, dil:%dpx, min:%d, probTh:%d)\n',scaleThrsh,erosionRadius,dilationRadius,minSize,probThresh);
                    outFinalImageMap = mergeUnetAndAll(smallScaleImagesMap, bigScaleImagesMap, sumProbMap, resizeFactor, scaleThrsh, probThresh, erosionRadius, dilationRadius, minSize);
                    
                    if evalAllResults
                        score = evaluation2(gtMap, outFinalImageMap);
                        fprintf('Score for corrected segmentations (scaleTh:%d, er:%dpx, dil:%dpx, min:%d, probTh:%d): %0.3f\n',scaleThrsh,erosionRadius,dilationRadius,minSize,probThresh,mean(cell2mat(score.values)));
                        fprintf(fid,'Score for corrected segmentations (scaleTh:%d, er:%dpx, dil:%dpx, min:%d, probTh:%d): %0.3f\n',scaleThrsh,erosionRadius,dilationRadius,minSize,probThresh,mean(cell2mat(score.values)));
                    end
                end
            end
        end
    end
end


%}

% FMINSEARCH
%x0 = double([14 12000 1 1 40 800]);
%fminsearch(@(x)optimizerFunc(x, gtMap, smallScaleImagesMap, bigScaleImagesMap, sumProbMap, fid), x0);

% FMINCON
%{
x0 = double([0.14 0.120 0.1 0.1 0.40]);
lb = double([0.05 0.100 0.0 0.1 0.05]);
ub = double([0.35 0.130 0.4 0.4 0.50]);
opts = optimset('Display','iter','Algorithm','interior-point', 'MaxIter', inf, 'MaxFunEvals', inf);
[x,fval,flag] = fmincon(@optimizerFunc, x0, [], [], [], [], lb, ub, [], opts);
%}

% For reproducibility
rng default 

% Start value and boundaries
% param vector values respectively:
% []
%x0 = [61, 8344,  1, 1, 37, 790]; => 0.5177 
%lb = [60,  7000, 0, 1, 20, 700, 20, 10,  9000];
%ub = [80, 12000, 2, 2, 30, 800, 40, 30, 11000];
%x0 = [67, 9900, 1, 1, 27, 750, 30, 21, 10000];

%scaleThrsh, probThresh, erosionRadius, dilationRadius, minSize, minOverlap, maxV, carea, median size
%lb = [30,  0, 0, 1, 10, 0000, 10, 10, 10];
%ub = [80, 32, 2, 2, 30, 0215, 30, 30, 30];
%x0 = [70, 12, 0, 1, 27, 0000, 30, 21, 20];

lb = [50, 15, 0, 1, 30, 0000, 20, 10, 10];
ub = [80, 32, 2, 2, 50, 0020, 40, 30, 30];
x0 = [75, 30, 1, 1, 41, 0012, 21, 17, 20];

%x0 = [60, 32, 0, 1, 0, 0000, 30, 21, 20];

%Apply randomed init values
%x0 = lb + rand(size(lb)).* (ub - lb);

% Int constraint for all variable
IntCon = 1:9;
popsize = 100;
gensize = 300;

% Set options and set the start values
options = gaoptimset('InitialPopulation', x0, 'PopulationSize', popsize, 'Generations', gensize, 'Display', 'iter', 'TolFun', 10e-1, 'TolCon', 10e-1);

%Optimize params
[x,fval,exitflag] = ga(@(x)optimizerFunc(x, gtMap, smallScaleImagesMap, bigScaleImagesMap, sumProbMap, fid), 9, [], [] ,[],[], lb, ub, [], IntCon, options);

%optimizerFunc(x0, gtMap, smallScaleImagesMap, bigScaleImagesMap, sumProbMap, fid);

fclose(fid);
delete(gcp('nocreate'));
