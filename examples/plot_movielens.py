import glob
import os

import imageio
imageio.plugins.ffmpeg.download()  # noqa

import matplotlib.pyplot as plt
import moviepy.editor as mpy
import numpy as np

from tensorrec import TensorRec
from tensorrec.eval import precision_at_k, recall_at_k
from tensorrec.loss_graphs import BalancedWMRBLossGraph
from tensorrec.prediction_graphs import DotProductPredictionGraph
from tensorrec.representation_graphs import NormalizedLinearRepresentationGraph

from test.datasets import get_movielens_100k

import logging
logging.getLogger().setLevel(logging.INFO)

# Load the movielens dataset
train_interactions, test_interactions, user_features, item_features, item_titles = \
    get_movielens_100k(negative_value=-1.0)

# Assemble parameters for fitting. 'epochs' is 1 in the fit_kwargs because we will be calling fit_partial 1000 times to
# run 1000 epochs.
epochs = 1000
fit_kwargs = {'epochs': 1, 'alpha': 0.0001, 'verbose': True, 'learning_rate': .01,
              'n_sampled_items': int(item_features.shape[0] * .1)}

# Build the TensorRec model
model = TensorRec(n_components=2,
                  biased=True,
                  loss_graph=BalancedWMRBLossGraph(),
                  prediction_graph=DotProductPredictionGraph(),
                  user_repr_graph=NormalizedLinearRepresentationGraph(),
                  normalize_users=True,
                  normalize_items=True,
                  n_tastes=3)

# Make some random selections of movies and users we want to plot
movies_to_plot = np.random.choice(a=item_features.shape[0], size=200, replace=False)
user_to_plot = np.random.choice(a=user_features.shape[0], size=400, replace=False)

# Iterate through 1000 epochs, outputting a JPG plot each epoch
for epoch in range(epochs):
    model.fit_partial(interactions=train_interactions, user_features=user_features, item_features=item_features,
                      **fit_kwargs)

    # The position of a movie or user is that movie's/user's 2-dimensional representation. The size of the movie dot is
    # related to its item bias.
    movie_positions = model.predict_item_representation(item_features)
    user_positions = model.predict_user_representation(user_features)
    movie_sizes = model.predict_item_bias(item_features) * 10 + 1.0

    # Handle multiple tastes, if applicable. If there are more than 1 taste per user, only the first of each user's
    # tastes will be plotted.
    if model.n_tastes > 1:
        user_positions = user_positions[0]

    _, ax = plt.subplots()
    ax.grid(b=True, which='both')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.scatter(*zip(*user_positions[user_to_plot]), color='r', s=1)
    ax.scatter(*zip(*movie_positions[movies_to_plot]), s=movie_sizes)
    ax.set_aspect('equal')

    for i, movie in enumerate(movies_to_plot):
        movie_name = item_titles[movie]
        movie_position = movie_positions[movie]
        # Uncomment this line to write movie titles to the plot.
        # ax.annotate(movie_name, movie_position[0:2], fontsize='x-small')

    file = '/tmp/tensorrec/movielens/epoch_{}.jpg'.format(epoch)
    plt.savefig(file)

    logging.info("Finished epoch {}".format(epoch))

p_at_k = precision_at_k(model, test_interactions,
                        user_features=user_features,
                        item_features=item_features,
                        k=5)
r_at_k = recall_at_k(model, test_interactions,
                     user_features=user_features,
                     item_features=item_features,
                     k=30)

logging.info("Precision@5: {}, Recall@30: {}".format(np.mean(p_at_k), np.mean(r_at_k)))

# Use the collected JPG files to create an MP4 video of the model fitting, then delete the JPGs.
fps = 12
file_list = glob.glob('/tmp/tensorrec/movielens/*.jpg')
list.sort(file_list, key=lambda x: int(x.split('_')[1].split('.jpg')[0]))
clip = mpy.ImageSequenceClip(file_list, fps=fps)
vid_file = '/tmp/tensorrec/movielens/movielens.mp4'
clip.write_videofile(filename=vid_file, fps=fps, codec='mpeg4', preset='veryslow', ffmpeg_params=['-qscale:v', '10'])
for file in file_list:
    os.remove(file)
