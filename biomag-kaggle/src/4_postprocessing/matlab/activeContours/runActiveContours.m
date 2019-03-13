function runActiveContours(inputDir,iter,tissueList,codePath,codeName)
% Runs active contour segmentation on images found in inputDir by each of
% their separated labelled mask (size of its bounding box) and writes
% output images. Handles images whose name is found in tissueList as
% "color" images with the corresponding settings of the program.
% inputDir: main folder of images as [inputDir]\[imageName]\...
% iter: number of iterations to run
% tissueList: full file name with path of the file containing image names
% of the tissue images.
% codePath: path of the compiled code (optional)
% codeName: name of the execuateble file to run (optional)
% 
% Example:
%     inputDir='d:\Letöltés\SZBK munka cuccok\kaggle\progress\2dseg\testing\';
%     iter=100;
%     tissueList='d:\Letöltés\SZBK munka cuccok\kaggle\progress\origImages\test_classified\Tissue_imageList.csv';
%     runActiveContours(inputDir,iter,tissueList);


%% run script
%system('"D:\DEV\VS\VS2015\2d-segmentation-gui\phasefieldGUIv2\x64\Release\phasefieldGUIv2PC.exe"  -nogui');

if nargin<4
    codePath='D:\KAGGLE_all\ActiveContour\Release\';
    cd(codePath);
    codePath=fullfile(codePath,'phasefieldGUIv2.exe');
elseif nargin==5
    cd(codePath);
    codePath=fullfile(codePath,codeName);
else
    fprintf('Please provide either of the following: 1) BOTH code path and name, or 2) NONE of them\n');
end

if exist(tissueList,'file')
    tissueList=importdata(tissueList);
else
    tissueList='';
end

dirs=dir(inputDir);
dirs=dirs([dirs.isdir]);
dirs=dirs(3:end);   % 1st and 2nd are '.' ans '..' dirs
% count images
counter=0;
for diri=1:numel(dirs)
    curDir=fullfile(inputDir,dirs(diri).name,'\GPU\');
    imagefiles = dir(fullfile(curDir,'*.png')); 
    counter=counter+length(imagefiles);
end
dones=0; t0=tic;
for diri=1:numel(dirs)
    curDir=fullfile(inputDir,dirs(diri).name,'\GPU\');
    imagefiles = dir(fullfile(curDir,'*.png'));      
    nfiles = length(imagefiles);    % Number of files found
%     iter = 100;  % 1250
    dirname =fullfile(inputDir,dirs(diri).name,'\out\');
    mkdir(dirname);
    t=tic;
    for ii=1:nfiles
        t1=tic;
       currentfilename = imagefiles(ii).name;
    %    currentimage = imread(fullfile(inputDir,currentfilename));
    %    images{ii} = currentimage;
    %    er =0;
    %    er= images{ii} >0;
    %    er*255;
    %    se = strel('disk',2);
    %    er=  imfill(er,'holes');
    %    er=imerode(er,se);
    %   imwrite(er,currentfilename-4,'png')

    if ~any(contains(tissueList,dirs(diri).name))
%         disp('gray');
        system([sprintf('"%s"',codePath) '  --nogui -i ',sprintf('"%s"',num2str(fullfile(curDir,currentfilename))),' -o ',sprintf('"%s"',num2str(fullfile(dirname,currentfilename))), ' -n ',num2str(iter)]);
    else
%         disp('color');
        % tissue image, use color version
        system([sprintf('"%s"',codePath) '  --nogui -i ',sprintf('"%s"',num2str(fullfile(curDir,currentfilename))),' -o ',sprintf('"%s"',num2str(fullfile(dirname,currentfilename))), ' -n ',num2str(iter),' -c']);
    end
    dones=dones+1;
%     state=ii/nfiles*100;
    state=dones/counter*100;
    disp([num2str(state) ' % done']);
    toc(t1);
    end
    fprintf('folder done in %f seconds\n',toc(t));
end
fprintf('all done in %f seconds\n',toc(t0));

end