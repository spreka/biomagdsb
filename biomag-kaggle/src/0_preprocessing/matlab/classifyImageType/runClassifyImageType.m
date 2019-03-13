function class=runClassifyImageType(src,srcM,dest,TSfile)
% Runs image type classification on images found in folder src and copies
% them by classes to dest, also copies masks to dest\masks; based on the
% training data (feature matrix) found in TSfile.
% 
% src='D:\Letöltés\SZBK munka cuccok\kaggle\progress\origImages\train\';
% srcM='d:\Letöltés\SZBK munka cuccok\kaggle\progress\masks\';
% dest='D:\Letöltés\SZBK munka cuccok\kaggle\progress\origImages\train_classified\';
% TSfile='D:\Letöltés\SZBK munka cuccok\kaggle\progress\saves\trainingSet.mat';

if ~exist(srcM,'dir')
    doMasks=false;
else
    doMasks=true;
end

destM=[dest 'masks\'];
mkdir([dest 'Fluorescent\']);
mkdir([dest 'Tissue\']);
mkdir([dest 'Brightfield\']);
if doMasks
    mkdir([destM 'Fluorescent\']);
    mkdir([destM 'Tissue\']);
    mkdir([destM 'Brightfield\']);
end

l=dir([src '*.png']);
c={'Fluorescent';'Tissue';'Brightfield'};
load(TSfile);

tic;
[class,posterior]=classifyImageType(l,trainingSet,labels,'classNames',c,'method','tree','batch',true);
toc;

for i=1:numel(l)
    idx=find(posterior(i,:)==1);
    if idx==1
        copyfile([src l(i).name],[dest 'Fluorescent\' l(i).name]);
        if doMasks copyfile([srcM l(i).name],[destM 'Fluorescent\' l(i).name]); end
    elseif idx==2
        copyfile([src l(i).name],[dest 'Tissue\' l(i).name]);
        if doMasks copyfile([srcM l(i).name],[destM 'Tissue\' l(i).name]); end
    elseif idx==3
        copyfile([src l(i).name],[dest 'Brightfield\' l(i).name]);
        if doMasks copyfile([srcM l(i).name],[destM 'Brightfield\' l(i).name]); end
    end
end

end