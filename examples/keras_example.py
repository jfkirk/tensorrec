import keras as ks

from tensorrec import TensorRec
from tensorrec.eval import fit_and_eval
from tensorrec.representation_graphs import AbstractKerasRepresentationGraph
from tensorrec.loss_graphs import BalancedWMRBLossGraph

from test.datasets import get_book_crossing

import logging
logging.getLogger().setLevel(logging.INFO)

train_interactions, test_interactions, user_features, item_features, _ = get_book_crossing(user_indicators=True,
                                                                                           item_indicators=True,
                                                                                           cold_start=True)


class ExampleKerasRepresentationGraph(AbstractKerasRepresentationGraph):
    def create_layers(self, n_features, n_components):
        return [
            ks.layers.Dense(n_components * 32, activation='relu'),
            ks.layers.Dense(n_components * 16, activation='relu'),
            ks.layers.Dense(n_components * 4, activation='relu'),
            ks.layers.Dense(n_components, activation='linear'),
        ]

model = TensorRec(n_components=10,
                  item_repr_graph=ExampleKerasRepresentationGraph(),
                  user_repr_graph=ExampleKerasRepresentationGraph(),
                  loss_graph=BalancedWMRBLossGraph())

fit_kwargs = {'epochs': 100, 'learning_rate': .01, 'n_sampled_items': 100, 'verbose': True}
result = fit_and_eval(model, user_features, item_features, train_interactions, test_interactions, fit_kwargs,
                      recall_k=100, precision_k=100, ndcg_k=100)

model.save_model('/tmp/tensorrec/keras_example/linear_graph')

logging.info(result)
