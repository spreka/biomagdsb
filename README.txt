--------- CITATION ---------
Please cite our paper if you use our method:
[...]


--------- PREREQUISITES ---------
Please see requirements.txt that can also be run as a bash script (Linux)
or alternatively, you can copy the install commands to console 
corresponding to your system (command prompt (Windows) / terminal (Linux))
and execute them.


Usage:

--------- PREDICTION WITH POST-PROCESSING ---------
Predicts nuclei first with a presegmenter Mask R-CNN model, estimates cell 
sizes, predcits with multiple U-Net models and ensembles the results, then
uses all of the above in a final post-processing step to refine the 
contours.

To predict nuclei on images plese do the following steps:

1. Please edit either 

	- start_prediction_full.bat (Windows) or 
	- start_prediction_full.sh (Linux)

and specify the following 3 directories with their corresponding full 
paths on your system:
	- Mask R-CNN: 	we provide a downloaded version that is compatible
			with our scripts; in the same folder as this file
			under Mask_RCNN-2.1 and it is the default folder
			used by the scripts. *WE STRONGLY RECOMMEND YOU
			USE THIS FOLDER* however, you can, if you will, 
			specify another folder where you have Mask R-CNN
			downloaded - in this case, we do not guarantee it
			will work.
	- root_dir:	full path of the folder containing this README. It
			is set automatically, but can be redefined.
	- images_dir:	full path of the folder containing your intensity
			images to segment. Default is the 'testImages' 
			folder we provide to test our method.

2. Open a command prompt (Windows) or terminal (Linux) and run 
start_prediction_full.[bat/sh] you just edited

3. Result of prediction will be found under the following relative path:
\kaggle_workflow\outputs\postprocessing\
relative to the folder containing this file.


--------- PREDICTION FAST ---------
Predicts nuclei with a presegmenter Mask R-CNN model that generalizes and
performs well in varying image types. Produces fast results that can be 
improved with the post-processing option above.

To predict fast:
Please follow the steps of "PREDICTION WITH POST-PROCESSING" section for
either of the files:

	- start_prediction_fast.bat (Windows) or 
	- start_prediction_fast.sh (Linux)

However, results will be on the following relative path:
\kaggle_workflow\outputs\presegment\


--------- CUSTOM VALIDATION USAGE ---------
To use your custom folder of images as validation please run the following
script according to your operating system:

	- runGenerateValidationCustom.bat (Windows)
	- runGenerateValidationCustom.sh (Linux)

If you wish to use our validation set assembled for Kaggle DSB, please 
rename the file relative to the directory of this file 
\matlab_scripts\generateValidation\validationFileNames_KAGGLE.mat
to \matlab_scripts\generateValidation\validationFileNames.mat


--------- TRAINING ---------
To train on your own images please follow the steps below.
WARNING: training will override the U-Net models we provide, we advise
you make a copy of them first from the following relative path:
\kaggle_workflow\unet\

1. Please run the following script according to your operating system:

	- start_training.bat (Windows)
	- start_training.sh (Linux)

	NOTE: for Windows you need to edit run_workflow_trainOnly.bat
	and set your python virtual environment path as indicated prior
	to running the script. It will open a second command prompt for
	necessary server running of pix2pix and must remain open until
	all pix2pix code execution is finished - which is indicated by
	the message "STYLE TRANSFER DONE:" in command prompt.

2. Resulting Mask R-CNN model will be found on the relative path:
\kaggle_workflow\outputs\model\mask_rcnn_trained.h5

(3.) If you would like to use this model for prediction, please copy
this file to the relative path
\kaggle_workflow\maskrcnn\model\mask_rcnn_final.h5
HOWEVER, it is possible to override our provided model of the same name;
we advise you make a copy of it prior to placing your trained model.

NOTE: in some cases Windows multi-threading is somewhat limited compared
to Unix-based systems which might result in the following error during 
pix2pix style transfer:
socket.error: [Errno 111] Connection refused
or
ConnectionRefusedError: [WinError 10061] No connection could be made 
because the target machine actively refused it
If you experience such, please rename the file
\biomag-kaggle\src\2_DL\style-transfer\pytorch-CycleGAN-and-pix2pix-etasnadi\options\base_options_win.py
to \biomag-kaggle\src\2_DL\style-transfer\pytorch-CycleGAN-and-pix2pix-etasnadi\options\base_options.py
(both paths are relative to the root folder of our code). We recommend
you make a copy of the original files before overwriting them. Further 
discussion of the issue can be found at
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/51
https://github.com/Lawouach/WebSocket-for-Python/issues/130
The original opntions file is:
\biomag-kaggle\src\2_DL\style-transfer\pytorch-CycleGAN-and-pix2pix-etasnadi\options\base_options_orig.py

If you have torch>0.4.1 installed a possible python error may arise when
executing pix2pix style transfer learning:
UserWarning: invalid index of a 0-dim tensor.
To overcome this, replace the file
\biomag-kaggle\src\2_DL\style-transfer\pytorch-CycleGAN-and-pix2pix-etasnadi\models\pix2pix_model.py
with pix2pix_model_newpytorch.py in the same folder. We recommend you
rename the original pix2pix_model.py first to preserve it.

--------- PARAMETER SEARCHING FOR POST-PROCESSING ---------
To find the most optimal parameters for your image set, please run the
following code. It will take only validation images to account.

	- start_parameterSearch.bat (Windows) or
	- start_parameterSearch.sh (Linux)

The output file will contain the found optimal parameters with their
corresponding mean IoU scores in the text file
\kaggle_workflow\outputsValidation\paramsearch\paramsearchresult.txt

The parameters can be passed to the post-processing function
\matlab_scripts\postProcess\postProcCodeRunnerFINAL.m
in either
	- run_workflow_predictOnly_full.bat (Windows) or
	- run_workflow_predictOnly_full.sh (Linux)
as its last parameter (default is "[]" which means empty array).
A description of the parameters can be found in the above .m file.

NOTE: by default the optimizer code runs for a time limit of 30 minutes.
If you would like to change the time limit, please edit either
	- run_workflow_parameterSearch4postProc.bat (Windows) or
	- run_workflow_parameterSearch4postProc.sh (Linux)
and set an additional input argument to the function 
"startPostProcParamSearch" with the time you desire passed in seconds.
For example "startPostProcParamSearch(rootDir,60)" will run the
optimizer for 60 seconds. If you would like to reset to the default,
just leave this parameter out as in "startPostProcParamSearch(rootDir)".


--------- PREPARE STYLE TRANSFER INPUT FOR SINGLE EXPERIMENT ---------
To prepare style transfer learning for your custom masks corresponding to
your test images please run either
	- start_singleExperimentPreparation.bat (Windows) or
	- start_singleExperimentPreparation.sh (Linux)

This will prepare training data structure for subsequent style transfer
learning steps for only one group containing all your test images.
After completing the preparation you should run either
	- start_training_singleExperiment.bat (Windows) or
	- start_training_singleExperiment.sh (Linux)
for training instead of start_training.bat/sh which performs style 
transfer learning based on image clustering.

WARNING: If you do not provide your own mask folder for this step the 
default option will be \kaggle_workflow\outputs\presegment which is
created by either
	- start_prediction_fast.bat (Windows) or 
	- start_prediction_fast.sh (Linux)

NOTE: This option should only be used if all your images come from the 
same experiment. If you provide mixed data, subsequent style transfer
learning will result in flawed models and failed synthetic images.
