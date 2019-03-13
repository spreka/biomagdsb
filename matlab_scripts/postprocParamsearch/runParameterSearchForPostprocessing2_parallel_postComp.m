function runParameterSearchForPostprocessing2_parallel_postComp(rootDir,timeLim)
%% postprocess script
% clear;

addpath(fullfile(rootDir,'biomag-kaggle/src/Utils/matlab'));
addpath(genpath('codes'));

gtFolder = fullfile(rootDir,'kaggle_workflow\inputs\validation\masks\');
inFolder = fullfile(rootDir,'kaggle_workflow\outputsValidation\maskrcnn\');
inProbmapFolders = {fullfile(rootDir,'kaggle_workflow/outputsValidation/ensemble/output/')};
outFolder = fullfile(rootDir,'kaggle_workflow/outputsValidation/paramsearch/');
resultsFile = fullfile(outFolder,'paramsearchresult.txt');
smallFolder='2x/';
largeFolder='4x/';


mkdir(outFolder);

fid = fopen(resultsFile, 'w');

smallScaleImagesFolder.scale = 1;
smallScaleImagesFolder.name = fullfile(inFolder, smallFolder, '');

bigScaleImagesFolder.scale = 2;
bigScaleImagesFolder.name = fullfile(inFolder, largeFolder, '');


%% settings
conn = 8;

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

toc


% For reproducibility
rng default 

% Start value and boundaries
% param vector values respectively:
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
defTimeLimit=1800;

if nargin<2
    timeLim=defTimeLimit;
end

% Set options and set the start values
options = gaoptimset('InitialPopulation', x0, 'PopulationSize', popsize, 'Generations', gensize, 'Display', 'iter', 'TolFun', 10e-1, 'TolCon', 10e-1 ...
    ,'TimeLimit', timeLim);

%Optimize params
[x,fval,exitflag] = ga(@(x)optimizerFunc(x, gtMap, smallScaleImagesMap, bigScaleImagesMap, sumProbMap, fid), 9, [], [] ,[],[], lb, ub, [], IntCon, options);

%optimizerFunc(x0, gtMap, smallScaleImagesMap, bigScaleImagesMap, sumProbMap, fid);

fclose(fid);
delete(gcp('nocreate'));

end