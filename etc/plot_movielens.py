import glob
import os

import imageio
imageio.plugins.ffmpeg.download()  # noqa

import matplotlib.pyplot as plt
import moviepy.editor as mpy
import numpy as np
from PIL import Image

from tensorrec import TensorRec
from tensorrec.eval import precision_at_k, recall_at_k
from tensorrec.loss_graphs import SeparationDenseLossGraph
from tensorrec.prediction_graphs import CosineDistancePredictionGraph

from test.datasets import get_movielens_100k

import logging
logging.getLogger().setLevel(logging.INFO)

train_interactions, test_interactions, user_features, item_features, item_titles = \
    get_movielens_100k(negative_value=-1.0)

epochs = 300
alpha = 0.0001
n_components = 2
biased = False
verbose = True
learning_rate = .01

compress_images = False

fit_kwargs = {'epochs': 1, 'alpha': alpha, 'verbose': verbose, 'learning_rate': learning_rate,
              'n_sampled_items': int(item_features.shape[0] * .01)}

model = TensorRec(n_components=n_components,
                  biased=biased,
                  loss_graph=SeparationDenseLossGraph(),
                  prediction_graph=CosineDistancePredictionGraph())

movies_to_plot = np.random.choice(a=item_features.shape[0], size=100, replace=False)
user_to_plot = np.random.choice(a=user_features.shape[0], size=200, replace=False)

for epoch in range(epochs):
    model.fit_partial(interactions=train_interactions, user_features=user_features, item_features=item_features,
                      **fit_kwargs)

    movie_positions = model.predict_item_representation(item_features)
    user_positions = model.predict_user_representation(user_features)

    _, ax = plt.subplots()
    ax.grid(b=True, which='both')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.scatter(*zip(*user_positions[user_to_plot]), color='r', s=1)
    ax.scatter(*zip(*movie_positions[movies_to_plot]))

    for i, movie in enumerate(movies_to_plot):
        movie_name = item_titles[movie]
        movie_position = movie_positions[movie]
        ax.annotate(movie_name, movie_position, fontsize='x-small')

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

logging.info("Precision:5: {}, Recall@30: {}".format(np.mean(p_at_k), np.mean(r_at_k)))

fps = 12
file_list = glob.glob('/tmp/tensorrec/movielens/*.jpg')
list.sort(file_list, key=lambda x: int(x.split('_')[1].split('.jpg')[0]))

if compress_images:
    logging.info('Compressing')
    for file in file_list:
        img = Image.open(file)
        img.save(file, 'JPEG',  quality=50)

clip = mpy.ImageSequenceClip(file_list, fps=fps)
vid_file = '/tmp/tensorrec/movielens/movielens.mp4'
clip.write_videofile(filename=vid_file, fps=fps, codec='mpeg4', preset='veryslow', ffmpeg_params=['-qscale:v', '10'])
for file in file_list:
    os.remove(file)
