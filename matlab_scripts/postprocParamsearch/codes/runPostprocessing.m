%% postprocess test
inMainFolder = 'd:\SZBK\kaggle\news\RCNN_results_0410\final_postProcd\';
outMainFolder = 'd:\SZBK\kaggle\news\RCNN_results_0410\final_postProcd\';

postProcessAllData([inMainFolder '4x_2x\'], [outMainFolder '2x\']);
postProcessAllData([inMainFolder '2x_2x\'], [outMainFolder '1x\']);
