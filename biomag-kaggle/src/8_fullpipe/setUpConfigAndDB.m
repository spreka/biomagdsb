function setUpConfigAndDB(maskFolder,targetDir,pretrainedClassifierPath,codeBase)

DB = FeatureDataBase();

DB.addFolderToDataBase(maskFolder,'tiff');

cellMaskDataBase = fullfile(targetDir,'DB.mat');
save(cellMaskDataBase,'DB');

pretrainedDistanceLearner = pretrainedClassifierPath;

save(fullfile(codeBase,'1_metalearning','config.mat'),'cellMaskDataBase','pretrainedDistanceLearner','codeBase');

end