# Parameters
# results_folder = Root folder for predicted outputs
# train = Folder for training data
# val = Folder for validation data
# test = Folder for images to predict
# epochs, model_name, batch, sigma = Do not change these
results_folder=/home/deeplearning/work/kaggle_workflow/unet/pretrained
train=/home/deeplearning/work/kaggle_workflow/unet/train
val=/home/deeplearning/work/kaggle_workflow/unet/train_eval
test=/home/deeplearning/work/kaggle_workflow/outputs

model1=$results_folder/UNet_forLassi_sigma0.0_1/UNet_forLassi_sigma0.0_1
model2=$results_folder/UNet_forLassi_sigma0.0_2/UNet_forLassi_sigma0.0_2
model3=$results_folder/UNet_forLassi_sigma1.0_1/UNet_forLassi_sigma1.0_1
model4=$results_folder/UNet_forLassi_sigma2.0_1/UNet_forLassi_sigma2.0_1
model5=$results_folder/UNet_forLassi_sigma0.5_1/UNet_forLassi_sigma0.5_1
model6=$results_folder/UNet_forLassi_sigma0.5_2/UNet_forLassi_sigma0.5_2

python train_sh.py --results_folder=$results_folder --train=$train --val=$val --test=$test --batch=1 --model_path=$model1 --model_name="UNet_forLassi_sigma0.0_1"
python train_sh.py --results_folder=$results_folder --train=$train --val=$val --test=$test --batch=1 --model_path=$model2 --model_name="UNet_forLassi_sigma0.0_2"
python train_sh.py --results_folder=$results_folder --train=$train --val=$val --test=$test --batch=1 --model_path=$model3 --model_name="UNet_forLassi_sigma1.0_1"
python train_sh.py --results_folder=$results_folder --train=$train --val=$val --test=$test --batch=1 --model_path=$model4 --model_name="UNet_forLassi_sigma2.0_1"
python train_sh.py --results_folder=$results_folder --train=$train --val=$val --test=$test --batch=1 --model_path=$model5 --model_name="UNet_forLassi_sigma0.5_1"
python train_sh.py --results_folder=$results_folder --train=$train --val=$val --test=$test --batch=1 --model_path=$model6 --model_name="UNet_forLassi_sigma0.5_2"
