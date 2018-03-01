from tensorrec import TensorRec
from tensorrec.eval import fit_and_eval
from tensorrec.loss_graphs import (
    RMSELossGraph, RMSEDenseLossGraph, SeparationLossGraph, SeparationDenseLossGraph, WMRBLossGraph
)
from tensorrec.prediction_graphs import DotProductPredictionGraph, CosineDistancePredictionGraph
from test.datasets import get_movielens_100k

import logging
logging.getLogger().setLevel(logging.INFO)

train_interactions, test_interactions, user_features, item_features = get_movielens_100k(negative_value=0)

epochs = 300
alpha = 0.00001
n_components = 10
biased = True
verbose = True
learning_rate = .1

n_sampled_items = int(item_features.shape[0] * .01)

fit_kwargs = {'epochs': epochs, 'alpha': alpha, 'verbose': verbose, 'learning_rate': learning_rate,
              'n_sampled_items': n_sampled_items}

res_strings = []

for loss_graph in (RMSELossGraph, RMSEDenseLossGraph, SeparationLossGraph, SeparationDenseLossGraph, WMRBLossGraph):
    for pred_graph in (DotProductPredictionGraph, CosineDistancePredictionGraph):

        model = TensorRec(n_components=n_components, biased=biased, loss_graph=loss_graph(),
                          prediction_graph=pred_graph())
        result = fit_and_eval(model, user_features, item_features, train_interactions, test_interactions, fit_kwargs)

        res_string = "{}".format(loss_graph.__name__)
        for _ in range(0, (30 - len(res_string))):
            res_string += " "
        res_string += "{}".format(pred_graph.__name__)
        for _ in range(0, (60 - len(res_string))):
            res_string += " "
        res_string += ": \t{} \t{}".format(result[0], result[1])

        print(res_string)
        res_strings.append(res_string)

print('--------------------------------------------------')
for res_string in res_strings:
    print(res_string)
