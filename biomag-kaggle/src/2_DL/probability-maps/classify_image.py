import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from utils import *

if __name__ == '__main__':
    model_name = 'UNet_augmented_data'
    all_datasets_path = '/media/tdanka/B8703F33703EF828/tdanka/data'
    train_dataset_loc = os.path.join(all_datasets_path, 'stage1_train_merged/loc.csv')
    test_dataset_loc = os.path.join(all_datasets_path, 'stage1_test/loc.csv')
    image_classes_loc = os.path.join(all_datasets_path, 'classes.csv')
    results_root_path = '/media/tdanka/B8703F33703EF828/tdanka/results'

    transform = make_single_transform(size=(256, 256), p_flip=0.5, color_jitter_params=(0.5, 0.5, 0.5, 0.5))
    train_dataset = ImageClassifierGenerator(train_dataset_loc, image_classes_loc, transform=transform)

    class_labels = ['FluorNormal', 'FluorLarge', 'FluorSmall', 'TissueSmall', 'TissueLarge', 'Brightfield']
    class_weights = [0.15, 0.5934065934065934, 0.5684210526315789, 0.782608695652174, 1.3846153846153846, 3.375]

