'''

Utility functions

'''

from os import path
from collections import OrderedDict


def split_batch(dic, size):
    batches = list()
    batch = OrderedDict()
    for idx, sample in enumerate(dic.items()):
        sample_id, class_values = sample
        if idx % size == 0 and len(batch) > 0:
            batches.append(batch)
            batch = OrderedDict()
        batch[sample_id] = class_values

    if len(batch) > 0:
        batches.append(batch)

    return batches


def split_batch(samples, size):
    batches = list()
    batch = list()
    for idx, sample in enumerate(samples):
        if idx % size == 0 and len(batch) > 0:
            batches.append(batch)
            batch = list()
        batch.append(sample)

    if len(batch) > 0:
        batches.append(batch)

    return batches


def split_ext_list(l):
    for idx, el in enumerate(l):
        l[idx] = path.splitext(el)[0]


def find_max_idx(arr):
    max = -1.0
    ret_idx = -1
    for idx, el in enumerate(arr):
        if el > max:
            ret_idx = idx
            max = el
    return ret_idx