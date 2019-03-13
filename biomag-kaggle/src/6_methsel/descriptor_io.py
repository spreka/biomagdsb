'''

Reads and writes CSV's (train/test sample descriptor files) in Abel's format.

'''


import csv
from collections import OrderedDict
from os import path


def read_descriptor_data(descriptor_file_name):
    class_names = list()
    samples = OrderedDict()

    # Load the labels for each image into  image paths and the image_class_labels
    # image_paths =         [im1, im2, ...imn]
    # image_class_labels =  [(lab1, lab2, ..., labk), (lab1, lab2, ..., labl), ...]
    with open(descriptor_file_name, newline='\n') as csvfile:
        descriptor_reader = csv.reader(csvfile, delimiter=',', quotechar='|', )
        for idx, row in enumerate(descriptor_reader):
            if idx == 0:
                class_names = row[1:]
            if idx > 0:
                image_id = path.splitext(row[0])[0]
                class_values = [float(v) for v in row[1:]]
                samples[image_id] = class_values

    return samples, class_names


def write_descriptor_data(descriptor_file_name, samples, class_names):
    # save the descriptor file
    with open(descriptor_file_name, 'w', newline='\n') as csv_file:
        descriptor_writer = csv.writer(csv_file, delimiter=',')
        descriptor_writer.writerow(['imageId', *class_names])
        for image_id, class_values in samples.items():
            descriptor_writer.writerow([image_id, *class_values])
