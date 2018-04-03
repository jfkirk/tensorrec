import keras as ks

from tensorrec import TensorRec
from tensorrec.eval import fit_and_eval
from tensorrec.representation_graphs import AbstractKerasRepresentationGraph
from tensorrec.loss_graphs import BalancedWMRBLossGraph

from test.datasets import get_book_crossing

import logging
logging.getLogger().setLevel(logging.INFO)

# Load the book crossing dataset
train_interactions, test_interactions, user_features, item_features, _ = get_book_crossing(user_indicators=True,
                                                                                           item_indicators=True,
                                                                                           cold_start=True)


# Construct a Keras representation graph by inheriting tensorrec.representation_graphs.AbstractKerasRepresentationGraph
class DeepRepresentationGraph(AbstractKerasRepresentationGraph):

    # This method returns an ordered list of Keras layers connecting the user/item features to the user/item
    # representation. When TensorRec learns, the learning will happen in these layers.
    def create_layers(self, n_features, n_components):
        return [
            ks.layers.Dense(n_components * 32, activation='relu'),
            ks.layers.Dense(n_components * 16, activation='relu'),
            ks.layers.Dense(n_components * 4, activation='relu'),
            ks.layers.Dense(n_components, activation='linear'),
        ]


# Use DeepRepresentationGraph for both item_repr and user_repr. This means that two separate deep neural nets will learn
# to process the input features - one will learn how to process items, and the other will learn to process users.
model = TensorRec(n_components=10,
                  item_repr_graph=DeepRepresentationGraph(),
                  user_repr_graph=DeepRepresentationGraph(),
                  loss_graph=BalancedWMRBLossGraph())

# Fit the model and get a result packet
fit_kwargs = {'epochs': 100, 'learning_rate': .01, 'n_sampled_items': 100, 'verbose': True}
result = fit_and_eval(model, user_features, item_features, train_interactions, test_interactions, fit_kwargs,
                      recall_k=100, precision_k=100, ndcg_k=100)
logging.info(result)
