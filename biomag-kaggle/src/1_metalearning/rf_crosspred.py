import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '..'))

from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader
from utils import *


def assemble_training_set(X, y, p_subsample=None):
    X_flat = X[:, :, :3].reshape(-1, 3)
    y_flat = y.reshape(-1, 1)
    if p_subsample:
        subsample_idx = np.random.choice(range(len(X_flat)), replace=False, size=int(len(X_flat)*p_subsample))
        return X_flat[subsample_idx], y_flat[subsample_idx]

    return X_flat, y_flat


data_path = '/home/namazu/Data/Kaggle/stage1_train_restructured'
loc_path = '/home/namazu/Data/Kaggle/train/loc.csv'

dataset = DatasetFromFolder(loc_path,transform=(lambda x: x/255))
X, y = next(iter(dataset))

X_train, y_train = assemble_training_set(X, y, 0.5)
