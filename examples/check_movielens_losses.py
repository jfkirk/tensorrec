from tensorrec import TensorRec
from tensorrec.eval import fit_and_eval
from tensorrec.representation_graphs import (
    LinearRepresentationGraph, ReLURepresentationGraph, NormalizedLinearRepresentationGraph
)
from tensorrec.loss_graphs import WMRBLossGraph, BalancedWMRBLossGraph
from tensorrec.prediction_graphs import (
    DotProductPredictionGraph, CosineSimilarityPredictionGraph, EuclideanSimilarityPredictionGraph
)
from tensorrec.util import append_to_string_at_point

from test.datasets import get_movielens_100k

import logging
logging.getLogger().setLevel(logging.INFO)

# Load the movielens dataset
train_interactions, test_interactions, user_features, item_features, _ = get_movielens_100k(negative_value=0)

# Construct parameters for fitting
epochs = 300
alpha = 0.00001
n_components = 10
verbose = True
learning_rate = .01
n_sampled_items = int(item_features.shape[0] * .1)
biased = False
fit_kwargs = {'epochs': epochs, 'alpha': alpha, 'verbose': verbose, 'learning_rate': learning_rate,
              'n_sampled_items': n_sampled_items}

res_strings = []

# Build results header
header = "Loss Graph"
header = append_to_string_at_point(header, 'Prediction Graph', 30)
header = append_to_string_at_point(header, 'ItemRepr Graph', 66)
header = append_to_string_at_point(header, 'Biased', 98)
header = append_to_string_at_point(header, 'N Tastes', 108)
header = append_to_string_at_point(header, 'Recall at 30', 120)
header = append_to_string_at_point(header, 'Precision at 5', 141)
header = append_to_string_at_point(header, 'NDCG at 30', 164)
res_strings.append(header)

# Iterate through many possibilities for model configuration
for loss_graph in (WMRBLossGraph, BalancedWMRBLossGraph):
    for pred_graph in (DotProductPredictionGraph, CosineSimilarityPredictionGraph,
                       EuclideanSimilarityPredictionGraph):
        for repr_graph in (LinearRepresentationGraph, ReLURepresentationGraph):
            for n_tastes in (1, 3):

                # Build the model, fit, and get a result packet
                model = TensorRec(n_components=n_components,
                                  n_tastes=n_tastes,
                                  biased=biased,
                                  loss_graph=loss_graph(),
                                  prediction_graph=pred_graph(),
                                  user_repr_graph=NormalizedLinearRepresentationGraph(),
                                  item_repr_graph=repr_graph())
                result = fit_and_eval(model, user_features, item_features, train_interactions, test_interactions,
                                      fit_kwargs)

                # Build results row for this configuration
                res_string = "{}".format(loss_graph.__name__)
                res_string = append_to_string_at_point(res_string, pred_graph.__name__, 30)
                res_string = append_to_string_at_point(res_string, repr_graph.__name__, 66)
                res_string = append_to_string_at_point(res_string, biased, 98)
                res_string = append_to_string_at_point(res_string, n_tastes, 108)
                res_string = append_to_string_at_point(res_string, ": {}".format(result[0]), 118)
                res_string = append_to_string_at_point(res_string, result[1], 141)
                res_string = append_to_string_at_point(res_string, result[2], 164)
                res_strings.append(res_string)
                print(res_string)

print('--------------------------------------------------')
for res_string in res_strings:
    print(res_string)
