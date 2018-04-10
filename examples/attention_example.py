from tensorrec import TensorRec
from tensorrec.eval import fit_and_eval
from tensorrec.representation_graphs import (
    LinearRepresentationGraph, NormalizedLinearRepresentationGraph
)
from tensorrec.loss_graphs import BalancedWMRBLossGraph

from test.datasets import get_movielens_100k

import logging
logging.getLogger().setLevel(logging.INFO)

# Load the movielens dataset
train_interactions, test_interactions, user_features, item_features, _ = get_movielens_100k(negative_value=0)

# Construct parameters for fitting
epochs = 500
alpha = 0.00001
n_components = 10
verbose = True
learning_rate = .01
n_sampled_items = int(item_features.shape[0] * .1)
fit_kwargs = {'epochs': epochs, 'alpha': alpha, 'verbose': verbose, 'learning_rate': learning_rate,
              'n_sampled_items': n_sampled_items}

# Build two models -- one without an attention graph, one with a linear attention graph
model_without_attention = TensorRec(
    n_components=10,
    n_tastes=3,
    user_repr_graph=NormalizedLinearRepresentationGraph(),
    attention_graph=None,
    loss_graph=BalancedWMRBLossGraph(),
)

model_with_attention = TensorRec(
    n_components=10,
    n_tastes=3,
    user_repr_graph=NormalizedLinearRepresentationGraph(),
    attention_graph=LinearRepresentationGraph(),
    loss_graph=BalancedWMRBLossGraph(),
)

results_without_attention = fit_and_eval(model=model_without_attention,
                                         user_features=user_features,
                                         item_features=item_features,
                                         train_interactions=train_interactions,
                                         test_interactions=test_interactions,
                                         fit_kwargs=fit_kwargs)
results_with_attention = fit_and_eval(model=model_with_attention,
                                      user_features=user_features,
                                      item_features=item_features,
                                      train_interactions=train_interactions,
                                      test_interactions=test_interactions,
                                      fit_kwargs=fit_kwargs)

logging.info("Results without attention: {}".format(results_without_attention))
logging.info("Results with attention:    {}".format(results_with_attention))
