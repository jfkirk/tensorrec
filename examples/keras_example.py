import keras as ks

from tensorrec import TensorRec
from tensorrec.eval import fit_and_eval, eval_random_ranks_on_dataset
from tensorrec.loss_graphs import BalancedWMRBLossGraph
from tensorrec.representation_graphs import (
    AbstractKerasRepresentationGraph, NormalizedLinearRepresentationGraph, LinearRepresentationGraph
)
from tensorrec.util import append_to_string_at_point

from test.datasets import get_book_crossing

import logging
logging.getLogger().setLevel(logging.INFO)

# Build results header string
result_strings = []
header = "UserRepr Graph"
header = append_to_string_at_point(header, 'ItemRepr Graph', 40)
header = append_to_string_at_point(header, 'Rec. In-Sample', 70)
header = append_to_string_at_point(header, 'Rec. Out-sample', 90)
header = append_to_string_at_point(header, 'Prec. In-Sample', 110)
header = append_to_string_at_point(header, 'Prec. Out-sample', 130)
header = append_to_string_at_point(header, 'NDCG In-Sample', 150)
header = append_to_string_at_point(header, 'NDCG Out-sample', 170)
result_strings.append(header)

# Load the book crossing dataset
train_interactions, test_interactions, user_features, item_features, _ = get_book_crossing(user_indicators=False,
                                                                                           item_indicators=True,
                                                                                           cold_start_users=True)

# Establish baseline metrics with random ranks for warm and cold start users
random_warm_result = eval_random_ranks_on_dataset(train_interactions, recall_k=100, precision_k=100, ndcg_k=100)
random_cold_result = eval_random_ranks_on_dataset(test_interactions, recall_k=100, precision_k=100, ndcg_k=100)
res_string = 'RANDOM BASELINE'
res_string = append_to_string_at_point(res_string, ": {:0.4f}".format(random_warm_result[0]), 68)
res_string = append_to_string_at_point(res_string, "{:0.4f}".format(random_cold_result[0]), 90)
res_string = append_to_string_at_point(res_string, "{:0.4f}".format(random_warm_result[1]), 110)
res_string = append_to_string_at_point(res_string, "{:0.4f}".format(random_cold_result[1]), 130)
res_string = append_to_string_at_point(res_string, "{:0.4f}".format(random_warm_result[2]), 150)
res_string = append_to_string_at_point(res_string, "{:0.4f}".format(random_cold_result[2]), 170)
logging.info(header)
logging.info(res_string)
result_strings.append(res_string)


# Construct a Keras representation graph by inheriting tensorrec.representation_graphs.AbstractKerasRepresentationGraph
class DeepRepresentationGraph(AbstractKerasRepresentationGraph):

    # This method returns an ordered list of Keras layers connecting the user/item features to the user/item
    # representation. When TensorRec learns, the learning will happen in these layers.
    def create_layers(self, n_features, n_components):
        return [
            ks.layers.Dense(n_components * 16, activation='relu'),
            ks.layers.Dense(n_components * 8, activation='relu'),
            ks.layers.Dense(n_components * 2, activation='relu'),
            ks.layers.Dense(n_components, activation='tanh'),
        ]


# Try different configurations using DeepRepresentationGraph for both item_repr and user_repr. If
# DeepRepresentationGraph is used, a deep neural network will learn to represent the users or items.
for user_repr in (NormalizedLinearRepresentationGraph, DeepRepresentationGraph):
    for item_repr in (LinearRepresentationGraph, DeepRepresentationGraph):
        model = TensorRec(n_components=20,
                          item_repr_graph=item_repr(),
                          user_repr_graph=user_repr(),
                          loss_graph=BalancedWMRBLossGraph(),
                          biased=False)

        # Fit the model and get a result packet
        fit_kwargs = {'epochs': 500, 'learning_rate': .01, 'n_sampled_items': 100, 'verbose': True}
        result = fit_and_eval(model, user_features, item_features, train_interactions, test_interactions, fit_kwargs,
                              recall_k=100, precision_k=100, ndcg_k=100)

        # Build results row for this configuration
        res_string = "{}".format(user_repr.__name__)
        res_string = append_to_string_at_point(res_string, item_repr.__name__, 40)
        res_string = append_to_string_at_point(res_string, ": {:0.4f}".format(result[3]), 68)
        res_string = append_to_string_at_point(res_string, "{:0.4f}".format(result[0]), 90)
        res_string = append_to_string_at_point(res_string, "{:0.4f}".format(result[4]), 110)
        res_string = append_to_string_at_point(res_string, "{:0.4f}".format(result[1]), 130)
        res_string = append_to_string_at_point(res_string, "{:0.4f}".format(result[5]), 150)
        res_string = append_to_string_at_point(res_string, "{:0.4f}".format(result[2]), 170)
        logging.info(header)
        logging.info(res_string)
        result_strings.append(res_string)

# Log the final results of all models
for res_string in result_strings:
    logging.info(res_string)
