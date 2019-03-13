%% postprocess script

gtFolder = 'd:\Projects\Data Science Bowl 2018\data\__ground-truth\out_stage1_validation_with_test\';

inFolder = 'd:\Projects\Data Science Bowl 2018\data\contest\20180407\data\segmentation\20180406_fulldata_Tensor1024_headLongTrain_test\';
inProbMapFolder = 'd:\Projects\Data Science Bowl 2018\data\__postprocessing\20180404_coco_resized\probmaps\stage1_test\';
outFolder = 'd:\Projects\Data Science Bowl 2018\data\contest\20180407\data\segmentation\20180406_fulldata_Tensor1024_headLongTrain_test\';
mkdir(outFolder);

%% settings
conn = 8;
% TODO test param
% minSize = 40;
% resize working image to original size
resizeFactor = 0.5;

saveMultiScaleImages = 1;
if saveMultiScaleImages
    masterScaleFolder = fullfile(outFolder, 'masterScale');
    mkdir(masterScaleFolder);
end


scalesFolders(1).scale = 1;
scalesFolders(1).name = fullfile(inFolder, 'Csabinak_1xtensor_2xim');

scalesFolders(2).scale = 2;
scalesFolders(2).name = fullfile(inFolder, 'Csabinak_2xtensor_2xim');

scalesFolders(3).scale = 4;
scalesFolders(3).name = fullfile(inFolder, 'Csabinak_4xtensor_2xim');

fileList = dir(fullfile(scalesFolders(1).name, '*.tiff'));

for i=1:numel(fileList)
    
    [~, fileBaseName,~] = fileparts(fileList(i).name)
    
    % collect segmentation from the proper tensor size results
    mergedImg = mergeScales(scalesFolders, fileList(i).name);
    mergedImg = imresize(mergedImg,resizeFactor,'nearest');
    
    if saveMultiScaleImages
        imwrite(mergedImg, fullfile(masterScaleFolder, fileList(i).name));
    end
    
    probMap = imread(fullfile(inProbMapFolder, [fileBaseName, '.png']));
    
    for erosionRadius = 1:3
        for dilationRadius = 1:3
            for minSize = [25:5:55]
                unetCorrectedImg = correctWithUnet(mergedImg, probMap, erosionRadius, dilationRadius);
                
                outFolderCaseName = sprintf('masterScale_e%d_d%d_min%d', erosionRadius, dilationRadius, minSize);
                outFolderCaseFullPath = fullfile(outFolder,outFolderCaseName);
                mkdir(outFolderCaseFullPath);
                
                % postprocessing
                outImg = postProcessKaggle(mergedImg, minSize, conn);
                
                % back to original size
                outImg = imresize(outImg, resizeFactor, 'nearest');
                
                % save to master dir
                imwrite(outImg, fullfile(outFolderCaseFullPath, fileList(i).name));
                
            end
        end
    end
end

evaluation(gtFolder, masterScaleFolder, masterScaleFolder, 'scores.csv');
for erosionRadius = 1:3
        for dilationRadius = 1:3
            for minSize = [25:5:55]
                
                outFolderCaseName = sprintf('masterScale_e%d_d%d_min%d', erosionRadius, dilationRadius, minSize);
                outFolderCaseFullPath = fullfile(outFolder,outFolderCaseName);
                
                evaluation(gtFolder, outFolderCaseFullPath, outFolderCaseFullPath, 'scores.csv');
            end
        end
end