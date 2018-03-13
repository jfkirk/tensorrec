from tensorrec import TensorRec
from tensorrec.eval import fit_and_eval
from tensorrec.representation_graphs import (
    LinearRepresentationGraph, ReLURepresentationGraph
)
from tensorrec.loss_graphs import (
    RMSEDenseLossGraph, SeparationDenseLossGraph, WMRBLossGraph,
)
from tensorrec.prediction_graphs import (
    DotProductPredictionGraph, CosineSimilarityPredictionGraph, EuclidianSimilarityPredictionGraph
)
from tensorrec.util import append_to_string_at_point

from test.datasets import get_movielens_100k

import logging
logging.getLogger().setLevel(logging.INFO)

train_interactions, test_interactions, user_features, item_features, _ = get_movielens_100k(negative_value=0)

epochs = 300
alpha = 0.00001
n_components = 10
verbose = True
learning_rate = .01

n_sampled_items = int(item_features.shape[0] * .1)

fit_kwargs = {'epochs': epochs, 'alpha': alpha, 'verbose': verbose, 'learning_rate': learning_rate,
              'n_sampled_items': n_sampled_items}

res_strings = []

header = "Loss Graph"
header = append_to_string_at_point(header, 'Prediction Graph', 30)
header = append_to_string_at_point(header, 'ItemRepr Graph', 66)
header = append_to_string_at_point(header, 'Biased', 98)
header = append_to_string_at_point(header, 'Recall at 30', 110)
header = append_to_string_at_point(header, 'Precision at 5', 131)
header = append_to_string_at_point(header, 'NDCG at 30', 154)
res_strings.append(header)

for biased in (True, False):
    for loss_graph in (RMSEDenseLossGraph, SeparationDenseLossGraph, WMRBLossGraph):
        for pred_graph in (DotProductPredictionGraph, CosineSimilarityPredictionGraph,
                           EuclidianSimilarityPredictionGraph):
            for repr_graph in (LinearRepresentationGraph, ReLURepresentationGraph):

                model = TensorRec(n_components=n_components,
                                  biased=biased,
                                  loss_graph=loss_graph(),
                                  prediction_graph=pred_graph(),
                                  user_repr_graph=LinearRepresentationGraph(),
                                  item_repr_graph=repr_graph())
                result = fit_and_eval(model, user_features, item_features, train_interactions, test_interactions,
                                      fit_kwargs)

                res_string = "{}".format(loss_graph.__name__)
                res_string = append_to_string_at_point(res_string, pred_graph.__name__, 30)
                res_string = append_to_string_at_point(res_string, repr_graph.__name__, 66)
                res_string = append_to_string_at_point(res_string, biased, 98)
                res_string = append_to_string_at_point(res_string, ": {}".format(result[0]), 108)
                res_string = append_to_string_at_point(res_string, result[1], 131)
                res_string = append_to_string_at_point(res_string, result[2], 154)

                print(res_string)
                res_strings.append(res_string)

print('--------------------------------------------------')
for res_string in res_strings:
    print(res_string)
