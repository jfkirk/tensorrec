from tensorrec import TensorRec
from tensorrec.eval import fit_and_eval
from tensorrec.representation_graphs import (
    LinearRepresentationGraph,
)
from tensorrec.loss_graphs import (
    RMSELossGraph, RMSEDenseLossGraph, SeparationLossGraph, SeparationDenseLossGraph, WMRBLossGraph
)
from tensorrec.prediction_graphs import (
    DotProductPredictionGraph, CosineDistancePredictionGraph, EuclidianDistancePredictionGraph
)
from tensorrec.util import append_to_string_at_point

from test.datasets import get_movielens_100k

import logging
logging.getLogger().setLevel(logging.INFO)

train_interactions, test_interactions, user_features, item_features, _ = get_movielens_100k(negative_value=0)

epochs = 300
alpha = 0.00001
n_components = 10
biased = True
verbose = True
learning_rate = .01

n_sampled_items = int(item_features.shape[0] * .01)

fit_kwargs = {'epochs': epochs, 'alpha': alpha, 'verbose': verbose, 'learning_rate': learning_rate,
              'n_sampled_items': n_sampled_items}

res_strings = []

header = "Loss Graph"
header = append_to_string_at_point(header, 'Prediction Graph', 30)
header = append_to_string_at_point(header, 'ItemRepr Graph', 64)
header = append_to_string_at_point(header, 'Recall at 30', 100)
header = append_to_string_at_point(header, 'Precision at 5', 122)
header = append_to_string_at_point(header, 'NDCG at 30', 146)
res_strings.append(header)

for loss_graph in (RMSELossGraph, RMSEDenseLossGraph, SeparationLossGraph, SeparationDenseLossGraph, WMRBLossGraph):
    for pred_graph in (DotProductPredictionGraph, CosineDistancePredictionGraph, EuclidianDistancePredictionGraph):
        for repr_graph in (LinearRepresentationGraph,):

            model = TensorRec(n_components=n_components, biased=biased,
                              loss_graph=loss_graph(),
                              prediction_graph=pred_graph(),
                              user_repr_graph=LinearRepresentationGraph(),
                              item_repr_graph=repr_graph())
            result = fit_and_eval(model, user_features, item_features, train_interactions, test_interactions,
                                  fit_kwargs)

            res_string = "{}".format(loss_graph.__name__)
            res_string = append_to_string_at_point(res_string, pred_graph.__name__, 30)
            res_string = append_to_string_at_point(res_string, repr_graph.__name__, 64)
            res_string = append_to_string_at_point(res_string, ": {}".format(result[0]), 98)
            res_string = append_to_string_at_point(res_string, result[1], 122)
            res_string = append_to_string_at_point(res_string, result[2], 146)

            print(res_string)
            res_strings.append(res_string)

print('--------------------------------------------------')
for res_string in res_strings:
    print(res_string)
