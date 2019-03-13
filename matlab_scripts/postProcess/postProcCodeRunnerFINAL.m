function postProcCodeRunnerFINAL(inMainFolder,outMainMAINFolder,scaleFolder1,...
    scaleFolder2,gtFolder,sizeSelectedFolder,probFolder,finalFolder,doEval,...
    origFolder,params)
% Calls runpostproc function with parsed inputs:
% inMainFolder: folder of mRCNN segmentations that contains 2x and 4x
%               outputs
% outMainMAINFolder: final output folder of postprocessing
% scaleFolder1: mRCNN segmentation output folder on 2x scale
% scaleFolder2: mRCNN segmentation output folder on 4x scale
% gtFolder: if the user wants to evaluate the results of post-processing
%           compared to ground truth (provided they have it), gtFolder is
%           its folder
% sizeSelectedFolder: temporary output folder of post-processing
% probFolder: folder of ensembled U-Net predictions
% finalFolder: temporary output folder of post-processing
% evalPath: if the user wants to evaluate the results of post-processing
%           compared to ground truth (provided they have it), this is the
%           path of the code for evaluation
% doEval: boolean flag to indicate whether the user wants evaluation
% origFolder: if the user wants to evaluate the results of post-processing
%             but ground truth folder is not avaiable/provided, this is the
%             folder containing the original images
% params: parameters of post-processing as follows: scaleThrsh, probThresh, 
%         erosionRadius, dilationRadius, minSize, minOverlap, maxV, carea, 
%         median size; 1-by-9 double vector or full path to a params file


if nargin==0
    % template
    inMainFolder = '/ssd/spreka/mrcnn_output/postprocessing/largeAug_0618_conf06_2XIMAGES/in/';
    outMainMAINFolder='/ssd/spreka/mrcnn_output/postprocessing/largeAug_0618_conf06_2XIMAGES_NEWunet/';
    scaleFolder1='C2x60/'; %'1x';
    scaleFolder2='C4x60/';

    gtFolder = '/ssd/spreka/mrcnn_input/stage2Test_tiff/';
    sizeSelectedFolder = 'master';
    probFolder = 'probmaps/ensembled/';
    finalFolder = 'masterBlaster';

    evalPath='/home/biomag/spreka/kaggle_new/biomag-kaggle/src/Utils/matlab';

    doEval=false;
    % if doEval is true and gtFolder is not parsed/ doesn't exist:
    origFolder='';
    
    % segmentation parameters:
    %scaleThrsh, probThresh, erosionRadius, dilationRadius, minSize, minOverlap, maxV, carea, median size
    params=[76, 32, 0, 1, 44, 0017, 32, 25, 21]; % optimal parameters
end

if ~exist('params','var') && (ischar(params) && exist(params,'file'))
    tmppar=importdata(params);
    params=tmppar.data';
elseif isempty(params)
    % use default values
    params=[76, 32, 0, 1, 44, 0017, 32, 25, 21]; % optimal parameters
end
ps={params};

% check gt folder existance
need2clearGT=false;
if ~exist(gtFolder,'dir')
    if ~exist(origFolder,'dir')
        fprintf('Cannot run post-processing because neither gtFolder nor origFolder exists\n');
        return;
    end
    mkdir(fullfile(outMainMAINFolder,'tmpGTreplacement'));
    need2clearGT=true;
    gtFolder=[fullfile(outMainMAINFolder,'tmpGTreplacement') filesep];
    exts={'png','tiff','tif','bmp','jpg','jpeg'};
    l=[];
    for e=1:numel(exts)
        l=[l;dir(fullfile(origFolder,['*.' exts{e}]))];
    end
    if ~isempty(l)
        for tmpimi=1:numel(l)
            [~,tmpbase,~]=fileparts(l(tmpimi).name);
            tmp=imread(fullfile(origFolder,l(tmpimi).name));
            s=size(tmp);
            imwrite(zeros(s(1),s(2),'uint8'),fullfile(outMainMAINFolder,'tmpGTreplacement',[tmpbase '.tiff']));
        end
    end
end


fprintf('\nstart postprocessing...\n');


outMainFolder = outMainMAINFolder;
if ~exist(outMainFolder,'dir') || isempty(dir(fullfile(outMainFolder,finalFolder,'*.tiff')))
    mkdir(outMainFolder);
    params=ps{1};

    postProcCodeFINAL(inMainFolder,outMainFolder,...
    scaleFolder1,scaleFolder2,gtFolder,sizeSelectedFolder,probFolder,...
    finalFolder,params);

    if doEval
        % print results
        addpath(evalPath);
        scores = evaluation(gtFolder, [outMainFolder finalFolder filesep], '', 'eval.csv');
        scoreMtx(1) = mean(cell2mat(scores.values));
        disp(['Best for ' outMainFolder ' = ' num2str(max(scoreMtx(1)))]);    
        rmpath(evalPath);
    end

else
    fprintf('%s already done\n',outMainFolder);
    if doEval
        % print results
        addpath(evalPath);
        scores = evaluation(gtFolder, [outMainFolder finalFolder filesep], '', 'eval.csv');
        scoreMtx(1) = mean(cell2mat(scores.values));
        disp(['Best for ' outMainFolder ' = ' num2str(max(scoreMtx(1)))]);    
        rmpath(evalPath);
    end
end

% clean up temporary result folders and move post-processed masks to output
% folder
moveList=dir(fullfile(outMainFolder,finalFolder,'*.tiff'));
for mi=1:numel(moveList)
    movefile(fullfile(outMainFolder,finalFolder,moveList(mi).name),fullfile(outMainFolder,moveList(mi).name));
end
rmdir(fullfile(outMainFolder,finalFolder),'s');
rmdir(fullfile(outMainFolder,sizeSelectedFolder),'s');
rmdir(fullfile(outMainFolder,scaleFolder1),'s');
rmdir(fullfile(outMainFolder,scaleFolder2),'s');

% clean up tmp GT replacement folder if it was needed be created
if need2clearGT
    rmdir(fullfile(outMainMAINFolder,'tmpGTreplacement'),'s');
end