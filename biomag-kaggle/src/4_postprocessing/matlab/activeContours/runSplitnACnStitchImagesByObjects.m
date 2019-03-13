function runSplitnACnStitchImagesByObjects(src,srcInt,dest,p,iter,tissueList,pad,gt,writeFlag,codePath,codeName)
% Runs postprocessing steps after segmentation to enhance contour quality
% as follows: splits images to small parts by each object's padded bounding
% box, applies active contour fitting to them and finally, stitches the
% small images together by the saved bounding box coordinates used in the
% first step.
% Parameters:
% src: folder of masks to split
% srcInt: folder of original intensity images
% dest: folder to write cropped images to as [dest]\[name]\out\[name*.png]
% p: padding size in pixels (for splitting images)
% forGPU: create structure for active contour GPU implementation (true)
% iter: number of iterations to run
% tissueList: full file name with path of the file containing image names
% of the tissue images.
% pad: padding size in pixels for stitching active contour results (remove
% padding!)
% gt: folder of ground truth images (for evaluation)
% writeFlag: whether to write evaluation scores to file in folder 'dest'
% codePath: path of the compiled code (optional)
% codeName: name of the execuateble file to run (optional)
% 
% Example1:
%     srcInt='D:\Letöltés\SZBK munka cuccok\kaggle\progress\origImages\test\';
%     src='d:\Letöltés\SZBK munka cuccok\kaggle\progress\probMaps_from_0223\UNet_multilabel_tissue+fluo\byclasses\test\ALLseeds\bbox_seeds\out_bbox_seeds_plus\';
%     p=5;
%     iter=100;
%     pad=40;
%     dest='d:\Letöltés\SZBK munka cuccok\kaggle\progress\2dseg\newTesting_nodil\';
%     tissueList='d:\Letöltés\SZBK munka cuccok\kaggle\progress\origImages\test_classified\Tissue_imageList.csv';
%     runSplitnACnStitchImagesByObjects(src,srcInt,dest,p,iter,tissueList,pad);
% 
% Example2:
%     srcInt='D:\Letöltés\SZBK munka cuccok\kaggle\progress\origImages\test\';
%     src='d:\Letöltés\SZBK munka cuccok\kaggle\progress\probMaps_from_0223\UNet_multilabel_tissue+fluo\byclasses\test\ALLseeds\bbox_seeds\out_bbox_seeds_plus\';
%     p=5;
%     iter=100;
%     pad=40;
%     dest='d:\Letöltés\SZBK munka cuccok\kaggle\progress\2dseg\newTesting_nodil\';
%     gt='';
%     tissueList='d:\Letöltés\SZBK munka cuccok\kaggle\progress\origImages\test_classified\Tissue_imageList.csv';
%     codePath='D:\KAGGLE_all\ActiveContour\Release\';
%     codeName='phasefieldGUIv2.exe';
%     runSplitnACnStitchImagesByObjects(src,srcInt,dest,p,iter,tissueList,pad,gt,true,codePath,codeName);
%
% Example3:
%     srcInt='D:\Letöltés\SZBK munka cuccok\kaggle\progress\origImages\test\';
%     src='d:\Letöltés\SZBK munka cuccok\kaggle\progress\probMaps_from_0223\UNet_multilabel_tissue+fluo\byclasses\test\ALLseeds\bbox_seeds\out_bbox_seeds_plus\';
%     p=5;
%     iter=100;
%     pad=40;
%     dest='d:\Letöltés\SZBK munka cuccok\kaggle\progress\2dseg\newTesting_nodil\';
%     gt='d:\Letöltés\SZBK munka cuccok\kaggle\progress\test_annotations_biomag\Annotation\all_merged\';
%     tissueList='d:\Letöltés\SZBK munka cuccok\kaggle\progress\origImages\test_classified\Tissue_imageList.csv';
%     runSplitnACnStitchImagesByObjects(src,srcInt,dest,p,iter,tissueList,pad,gt,true);

evalFlag=true;
if nargin<10
    % use default active contour path and name
    codePath='D:\DEV\VS\VS2015\2d-segmentation-gui\phasefieldGUIv2\x64\Release\';
    codeName='phasefieldGUIv2PC.exe';
    if nargin<8 || ~exist(gt,'dir')
        % no need to evaluate
        evalFlag=false;
    end
end

forGPU=true;
% split images to [dest]\[name]\images\[name*.png] and
% [dest]\[name]\masks\[name*.png], copy correct structure to
% [dest]\[name]\GPU\[name*.png] and \[name*.png.mask] for active contour
% fitting in the next step
splitImagesByObjects(src,srcInt,dest,p,forGPU);
% run active contours and write results to [dest]\[name]\out\[name*.png]
runActiveContours(dest,iter,tissueList,codePath,codeName)
% stitch these results to images of original size again
stitchImagesByObjects(dest,pad);

if evalFlag
    scores=runEval(dest,gt,writeFlag);
    fprintf('\nMean score: %f\n',mean(scores));
end

end