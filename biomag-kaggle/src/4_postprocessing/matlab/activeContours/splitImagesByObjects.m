function splitImagesByObjects(src,srcInt,dest,p,forGPU)
% Runs image splitting by labelled masks' bounding boxes for all images
% found in folder 'src'.
% src: folder of masks to split
% srcInt: folder of original intensity images
% dest: folder to write cropped images to
% p: padding size in pixels
% forGPU: create structure for active contour GPU implementation
%
% Example:
%     srcInt='D:\Letöltés\SZBK munka cuccok\kaggle\progress\origImages\test\';
%     dest='d:\Letöltés\SZBK munka cuccok\kaggle\progress\2dseg\testing\';
%     src='D:\Letöltés\SZBK munka cuccok\kaggle\progress\others_results\resultsRCNN\20180317_Hist\stage1_test_w_all_model\';
%     p=5;
%     forGPU=true;
%     splitImagesByObjects(src,srcInt,dest,p,forGPU);

if nargin<5
    forGPU=false;
    if nargin<4
        p=5;
    end
end

if ~exist(dest,'dir')
    mkdir(dest);
end

l=dir(fullfile(src,'*.png'));
if isempty(l)
    l=dir(fullfile(src,'*.tiff'));
    if isempty(l)
        l=dir(fullfile(src,'*.tif'));
    end
end

fprintf('slitting images by object masks...\n'); t0=tic;
for i=1:numel(l)
    t=tic;
    [~,imName,~]=fileparts(l(i).name);
    intName=fullfile(srcInt,[imName '.png']);
    if ~exist(intName,'file')
        intName=fullfile(srcInt,[imName '.tiff']);
        if ~exist(intName,'file')
            intName=fullfile(srcInt,[imName '.tif']);
        end
    end
    intImage=imread(intName);
    maskImage=imread(fullfile(src,l(i).name));
    splitImageByObjects(intImage,maskImage,dest,l(i).name,p,forGPU);
    toc(t);
end
fprintf('finished in %f seconds\n',toc(t0));
end

function bboxSize=splitImageByObjects(intImage,maskImage,dest,name,p,forGPU)
% Splits both original intensity-image and labelled mask image by the
% bounding boxes of each object found on the mask.
% intImage: original intensity image
% maskImage: labelled mask image
% dest: folder to write the cropped images to
% name: base name of the images to write

if nargin<6
    forGPU=false;
end

[~,name,~]=fileparts(name);
if ~exist(fullfile(dest,name,'masks'),'dir')
    mkdir(fullfile(dest,name,'masks'));
end
if ~exist(fullfile(dest,name,'images'),'dir')
    mkdir(fullfile(dest,name,'images'));
end
if forGPU
    if ~exist(fullfile(dest,name,'GPU'),'dir')
        mkdir(fullfile(dest,name,'GPU'));
    end
end
s=size(maskImage);
% p=5;    % padding pixel size
reg=regionprops(maskImage,'BoundingBox');
bboxSize=zeros(numel(reg),4);
for i=1:numel(reg)
    bbox=reg(i).BoundingBox;
    bboxSize(i,:)=[max(1,bbox(1)-p) max(1,bbox(2)-p) min(s(1),bbox(3)+2*p) min(s(2),bbox(4)+2*p)];
    croppedMask=imcrop(maskImage==i,bboxSize(i,:));
    croppedGray=imcrop(intImage,bboxSize(i,:));
%     newname=sprintf('%s_%03d.png',name,i);  % this creates too long file names!
    newname=sprintf('%03d.png',i);
    imwrite(croppedMask,fullfile(dest,name,'masks',newname));
    imwrite(croppedGray,fullfile(dest,name,'images',newname));
    if forGPU
        copyfile(fullfile(dest,name,'images',newname),fullfile(dest,name,'GPU',newname));
        copyfile(fullfile(dest,name,'masks',newname),fullfile(dest,name,'GPU',[newname '.mask']));
    end
end
save(fullfile(dest,[name '_bbox.mat']),'name','bboxSize','s','p');
end