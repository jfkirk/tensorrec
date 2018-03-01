import matplotlib.pyplot as plt

from tensorrec import TensorRec
from tensorrec.eval import fit_and_eval
from tensorrec.loss_graphs import SeparationDenseLossGraph
from tensorrec.prediction_graphs import DotProductPredictionGraph
from tensorrec.util import append_to_string_at_point

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

fit_kwargs = {'epochs': epochs, 'alpha': alpha, 'verbose': verbose, 'learning_rate': learning_rate}

model = TensorRec(n_components=n_components, biased=biased, loss_graph=SeparationDenseLossGraph(),
                  prediction_graph=DotProductPredictionGraph())
result = fit_and_eval(model, user_features, item_features, train_interactions, test_interactions, fit_kwargs)

res_string = "{}".format(model.loss_graph_factory.__class__.__name__)
res_string = append_to_string_at_point(res_string, model.prediction_graph_factory.__class__.__name__, 30)
res_string = append_to_string_at_point(res_string, ": {}".format(result[0]), 64)
res_string = append_to_string_at_point(res_string, result[1], 88)

print(res_string)

movie_positions = model.predict_item_representation(item_features)

movies_to_plot = (100, 200)

fig, ax = plt.subplots()
ax.grid(b=True, which='both')
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
ax.scatter(*zip(*movie_positions[movies_to_plot[0]:movies_to_plot[1]]))

for i, movie_name in enumerate(item_titles[movies_to_plot[0]:movies_to_plot[1]]):
    ax.annotate(movie_name, movie_positions[i + movies_to_plot[0]], fontsize='x-small')
fig.show()
