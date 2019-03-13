'''

This script gathers parts of the dataset into one directory that can be useful when training augmentation
is needed. One should prepare the augmented data into a different training set from the original one
and these training data and their descriptors are then merged together into a simple target directory and
descriptor file.

Input:
    training_data_paths = ['path#1', 'path#2', ..., 'path#k']
    training_descriptors = ['path#1.csv', 'path#2.csv', ..., 'path#k.csv']

    training_data_path = 'output_path'
    training_data_descriptor = 'output_file.csv'

Output:
    the merged training data and descriptor file

'''

from os import path
import shutil as sh
from descriptor_io import read_descriptor_data
from descriptor_io import write_descriptor_data
from collections import OrderedDict


def merge_training_sets(data_paths, descriptor_files, result_data_path, result_descriptor_file):
    # Merge the data of the sets by copying and merge the descriptors
    if len(descriptor_files) > 0:
        _, class_names = read_descriptor_data(descriptor_files[0])
        merged_samples = OrderedDict()
        for data_set_idx, desc_file in enumerate(descriptor_files):
            samples, _ = read_descriptor_data(desc_file)
            merged_samples.update(samples)
            data_path = data_paths[data_set_idx]
            for image_id in samples.keys():
                dir_src = path.join(data_path, image_id)
                dir_dst = path.join(result_data_path, image_id)
                sh.copytree(dir_src, dir_dst)
        # Put the result to a csv
        write_descriptor_data(result_descriptor_file, merged_samples, class_names)

data_paths = [
    '/home/etasnadi/dev/kaggle-dataset/mergetest1',
    '/home/etasnadi/dev/kaggle-dataset/mergetest2'
]
descriptor_files = [
    '/home/etasnadi/dev/kaggle-dataset/mergetest1/desc.csv',
    '/home/etasnadi/dev/kaggle-dataset/mergetest2/desc.csv'
]
result_data_path = '/home/etasnadi/dev/kaggle-dataset/merged'
result_descriptor_file = '/home/etasnadi/dev/kaggle-dataset/merged/desc.csv'

merge_training_sets(data_paths, descriptor_files, result_data_path, result_descriptor_file)
