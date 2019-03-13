# Parameters
# results_folder = Root folder for predicted outputs
# train = Folder for training data
# val = Folder for validation data
# test = Folder for images to predict
# epochs, model_name, batch, sigma = Do not change these
results_folder=/media/tdanka/B8703F33703EF828/lassi/results/shell_test
train=/media/tdanka/B8703F33703EF828/lassi/data/forLassi_split/train
val=/media/tdanka/B8703F33703EF828/lassi/data/forLassi_split/val
test=/media/tdanka/B8703F33703EF828/lassi/data/stage1_test_collected
epochs=100
batch=12
python train_sh.py --results_folder=$results_folder --train=$train --val=$val --test=$test --epochs=$epochs --batch=$batch --model_name=UNet_sigma0.0_1 --sigma=0.0
python train_sh.py --results_folder=$results_folder --train=$train --val=$val --test=$test --epochs=$epochs --batch=$batch --model_name=UNet_sigma0.0_2 --sigma=0.0
python train_sh.py --results_folder=$results_folder --train=$train --val=$val --test=$test --epochs=$epochs --batch=$batch --model_name=UNet_sigma0.5_1 --sigma=0.5
python train_sh.py --results_folder=$results_folder --train=$train --val=$val --test=$test --epochs=$epochs --batch=$batch --model_name=UNet_sigma0.5_2 --sigma=0.5
python train_sh.py --results_folder=$results_folder --train=$train --val=$val --test=$test --epochs=$epochs --batch=$batch --model_name=UNet_sigma1.0_1 --sigma=1.0
python train_sh.py --results_folder=$results_folder --train=$train --val=$val --test=$test --epochs=$epochs --batch=$batch --model_name=UNet_sigma2.0_1 --sigma=2.0
