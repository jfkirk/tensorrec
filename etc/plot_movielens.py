import glob
import os

import imageio
imageio.plugins.ffmpeg.download()
import matplotlib.pyplot as plt
import moviepy.editor as mpy

from tensorrec import TensorRec
from tensorrec.loss_graphs import WMRBLossGraph
from tensorrec.prediction_graphs import DotProductPredictionGraph

from test.datasets import get_movielens_100k

import logging
logging.getLogger().setLevel(logging.INFO)

train_interactions, test_interactions, user_features, item_features, item_titles = \
    get_movielens_100k(negative_value=-1.0)

epochs = 300
alpha = 0.00001
n_components = 2
biased = False
verbose = True
learning_rate = .01

fit_kwargs = {'epochs': 1, 'alpha': alpha, 'verbose': verbose, 'learning_rate': learning_rate,
              'n_sampled_items': int(item_features.shape[0] * .01)}

model = TensorRec(n_components=n_components, biased=biased, loss_graph=WMRBLossGraph(),
                  prediction_graph=DotProductPredictionGraph())

for epoch in range(epochs):
    model.fit_partial(interactions=train_interactions, user_features=user_features, item_features=item_features,
                      **fit_kwargs)

    movie_positions = model.predict_item_representation(item_features)

    movies_to_plot = (100, 200)

    _, ax = plt.subplots()
    ax.grid(b=True, which='both')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.scatter(*zip(*movie_positions[movies_to_plot[0]:movies_to_plot[1]]))

    for i, movie_name in enumerate(item_titles[movies_to_plot[0]:movies_to_plot[1]]):
        ax.annotate(movie_name, movie_positions[i + movies_to_plot[0]], fontsize='x-small')
    plt.savefig('/tmp/tensorrec/movielens/epoch_{}.png'.format(epoch))
    logging.info("Finished epoch {}".format(epoch))

fps = 12
file_list = glob.glob('/tmp/tensorrec/movielens/*.png') # Get all the pngs in the current directory
list.sort(file_list, key=lambda x: int(x.split('_')[1].split('.png')[0])) # Sort the images by #, this may need to be tweaked for your use case
clip = mpy.ImageSequenceClip(file_list, fps=fps)
clip.write_gif('/tmp/tensorrec/movielens/movielens.gif', fps=fps)
for file in file_list:
    os.remove(file)
