%list folders to add to the db on the local machine

load('config.mat');

DB = FeatureDataBase();
% {
f = 'e:\kaggle\Data\feri_All_train_data_20180411_0106\masks\';
DB.addFolderToDataBase(f,'tiff');

%{
f = '/home/biomag/szkabel/cellMasks/out_histo_mergedMaskLabelledImage';
DB.addFolderToDataBase(f);

f = '/home/biomag/szkabel/cellMasks/outstage1_mergedMaskLabelledImage';
DB.addFolderToDataBase(f);
%}

save(cellMaskDataBase,'DB');


%{
f = 'd:\�bel\SZBK\Projects\Kaggle\Data\Annotations\Merged\fluo_mergedMaskLabelledImage\';
DB.addFolderToDataBase(f);

f = 'd:\�bel\SZBK\Projects\Kaggle\Data\Annotations\Merged\fluo2_mergedMaskLabelledImage\';
DB.addFolderToDataBase(f);

f = 'd:\�bel\SZBK\Projects\Kaggle\Data\Annotations\Merged\histo_mergedMaskLabelledImage\';
DB.addFolderToDataBase(f);

f = 'd:\�bel\SZBK\Projects\Kaggle\Data\Annotations\Merged\histo2_mergedMaskLabelledImage\';
DB.addFolderToDataBase(f);

f = 'd:\�bel\SZBK\Projects\Kaggle\Data\Annotations\Merged\stage1_test_mergedMaskLabelledImage\';
DB.addFolderToDataBase(f);

f = 'd:\�bel\SZBK\Projects\Kaggle\Data\Annotations\Merged\stage1_train_mergedMaskLabelledImage\';
DB.addFolderToDataBase(f);
%}