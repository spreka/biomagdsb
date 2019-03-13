import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from dataset import chk_mkdir

from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool


def float_to_RGBHEX(x):
    assert 0 <= x <= 1, 'x must be between 0 and 1'
    gray_int = int(x*240)
    return "#%02x%02x%02x" % (gray_int, gray_int, gray_int)


# csv paths
data_loc_path = '/media/tdanka/B8703F33703EF828/tdanka/data/stage1_train_merged/loc.csv'
features_path = '/media/tdanka/B8703F33703EF828/tdanka/data/train_image_stats.csv'
scores_path = '/media/tdanka/B8703F33703EF828/tdanka/data/mergedTrainScores.csv'

# export paths
plots_path = '/media/tdanka/B8703F33703EF828/tdanka/plots'
chk_mkdir(plots_path)

# features
features = pd.read_csv(features_path, header=0)
features['ImageName'] = features['ImageName'].apply(lambda x: x[:-4])
features = features.rename(columns={'ImageName': 'name'}).sort_values(by='name')

# scores
scores = pd.read_csv(scores_path, header=0)
scores['ImageName'] = scores['ImageName'].apply(lambda x: x[:-4])
scores = scores.rename(columns={'ImageName': 'name'}).sort_values(by='name')

# PCA on features
val = features.loc[:, features.columns != 'name'].values
val_pca = PCA(n_components=2).fit_transform(val)
# computing ranges
x_min, x_max = np.min(val_pca[:, 0]), np.max(val_pca[:, 0])
y_min, y_max = np.min(val_pca[:, 1]), np.max(val_pca[:, 1])

# KDE on PCA
kde = KernelDensity(kernel='gaussian', bandwidth=0.25)
kde.fit(val_pca)
image_density = kde.score_samples(val_pca)

for method_name in scores.columns:
    if method_name != 'name':
        # creating dataframe for plotting
        plot_df = pd.concat([features, pd.DataFrame(val_pca, index=features.index, columns=['x', 'y'])], axis=1)[['name', 'x', 'y']]
        plot_df['location'] = plot_df['name'].apply(lambda x: 'train/'+x+'.png')

        # adding colors
        plot_df['color'] = pd.DataFrame([float_to_RGBHEX(val) for val in scores[method_name]], index=plot_df.index)
        # adding scores
        plot_df = plot_df.merge(scores[[method_name, 'name']].rename(columns={method_name: 'score'}), on='name')

        output_file(os.path.join(plots_path, method_name+'_big.html'))

        source = ColumnDataSource(plot_df)

        hover = HoverTool(
            tooltips=
            """
            name: @name <br>
            score: @score <br>
            <img src="@location" height="200">
            """
        )

        p = figure(
            plot_width=2000, plot_height=2000, x_range=[x_min-1, x_max+1], y_range=[y_min-1, y_max+1],
            tools=[hover], title="Scores with " + method_name
        )

        p.circle(x='x', y='y', size=10, color='color', source=source)

        show(p)