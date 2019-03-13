import os
import glob_vars


def read_db(dataset_dir):
    clusters = os.listdir(os.path.join(dataset_dir, glob_vars.IMAGE_CLUSTERS_REL_DIR))
    DB = dict()
    for cluster in clusters:
        DB[cluster] = os.listdir(os.path.join(dataset_dir, glob_vars.IMAGE_CLUSTERS_REL_DIR, cluster))
    return DB