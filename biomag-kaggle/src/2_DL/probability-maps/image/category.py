import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize
from torch.utils.data import Dataset
from sklearn.random_projection import SparseRandomProjection, johnson_lindenstrauss_min_dim
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool


def rgb2hex(r, g, b):
    return "#%02x%02x%02x" % (r, g, b)


class ImageGenerator(Dataset):

    def __init__(self, im_paths, flatten=False, with_loc=False):
        self.im_paths = pd.read_csv(im_paths, index_col=[0])
        self.flatten = flatten
        self.with_loc = with_loc

    def __len__(self):
        return len(self.im_paths)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= len(self.im_paths):
            raise StopIteration
        else:
            self.idx += 1
            return self.__getitem__(self.idx-1)

    def __getitem__(self, idx):
        if idx > len(self.im_paths):
            raise StopIteration

        # getting the image and the mask
        image = io.imread(self.im_paths.loc[idx, 'image_path'])
        image = rgb2gray(image[:, :, :3])

        # getting the filename
        name = self.im_paths.loc[idx, 'name']

        image = resize(image, output_shape=(256, 256))

        mean = np.mean(image)
        var = np.std(image)

        stat = np.array([mean, var])

        if self.flatten:
            image = image.reshape(1, -1)

        if self.with_loc:
            return image, stat, name, self.im_paths.loc[idx, 'image_path']
        else:
            return image, stat, name


if __name__ == '__main__':
    path = {
        'train': '/media/tdanka/B8703F33703EF828/tdanka/data/stage1_train_merged/loc.csv',
        'test': '/media/tdanka/B8703F33703EF828/tdanka/data/stage1_test/loc.csv'
    }

    dataset = {
        'train': ImageGenerator(path['train'], flatten=True, with_loc=True),
        'test': ImageGenerator(path['test'], flatten=True, with_loc=True)
    }

    stats = {'train': list(), 'test': list()}

    color = {'train': rgb2hex(0, 0, 255), 'test': rgb2hex(255, 0, 0)}

    for name in ['train', 'test']:
        for image_idx, (image, image_stat, image_name, image_loc) in enumerate(iter(dataset[name])):
            stats[name].append((image_name, *list(image_stat), image_loc, color[name]))

        stats[name] = pd.DataFrame(
            stats[name],
            columns=['name', 'mean', 'std', 'image_path', 'color']
        )

    plots_path = '/media/tdanka/B8703F33703EF828/tdanka/plots'
    output_file(os.path.join(plots_path, 'vis.html'))
    plot_df = pd.concat([stats['train'], stats['test']])
    source = ColumnDataSource(plot_df)

    hover = HoverTool(
        tooltips=
        """
        name: @name <br>
        <img src="@image_path" height="200">
        """
    )

    p = figure(plot_width=1000, plot_height=1000, tools=[hover])

    p.circle(x='mean', y='std', size=10, color='color', source=source)

    show(p)

    """
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(15, 15))
        plt.scatter(stats['train'].iloc[:, 0], stats['train'].iloc[:, 1], c='b')
        plt.scatter(stats['test'].iloc[:, 0], stats['test'].iloc[:, 1], c='r')
        plt.show()
    """
