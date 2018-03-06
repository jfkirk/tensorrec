import keras as ks

from tensorrec import TensorRec
from tensorrec.eval import fit_and_eval
from tensorrec.representation_graphs import AbstractKerasRepresentationGraph
from tensorrec.loss_graphs import SeparationDenseLossGraph

from test.datasets import get_movielens_100k

import logging
logging.getLogger().setLevel(logging.INFO)

train_interactions, test_interactions, user_features, item_features, _ = get_movielens_100k()


class ExampleKerasRepresentationGraph(AbstractKerasRepresentationGraph):
    def create_layers(self, n_features, n_components):
        return [
            ks.layers.Dense(int(n_features / 2), activation='relu'),
            ks.layers.Dense(n_components * 2, activation='relu'),
            ks.layers.Dense(n_components, activation='tanh'),
        ]

model = TensorRec(n_components=10,
                  item_repr_graph=ExampleKerasRepresentationGraph(),
                  loss_graph=SeparationDenseLossGraph())

fit_kwargs = {'epochs': 1000, 'learning_rate': .001, 'verbose': True}
result = fit_and_eval(model, user_features, item_features, train_interactions, test_interactions, fit_kwargs)

logging.info(result)
