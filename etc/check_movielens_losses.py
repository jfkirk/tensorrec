from tensorrec import TensorRec
from tensorrec.eval import fit_and_eval
from tensorrec.loss_graphs import RMSEDenseLossGraph, SeparationLossGraph, SeparationDenseLossGraph, WMRBLossGraph, \
    WMRBAlignmentLossGraph
from test.datasets import get_movielens_100k

import logging
logging.getLogger().setLevel(logging.INFO)

train_interactions, test_interactions, user_features, item_features = get_movielens_100k(negative_value=0)

epochs = 300
alpha = 0.00001
n_components = 10
biased = False
verbose = True
learning_rate = .1

n_sampled_items = int(item_features.shape[0] * .01)

fit_kwargs = {'epochs': epochs, 'alpha': alpha, 'verbose': verbose, 'learning_rate': learning_rate,
              'n_sampled_items': n_sampled_items}

model_baseline = TensorRec(n_components=n_components, biased=biased)
rmse_result = fit_and_eval(model_baseline, user_features, item_features, train_interactions, test_interactions,
                           fit_kwargs)
print("RMSE: \t\t%s, \t%s" % rmse_result)

model_baseline = TensorRec(n_components=n_components, biased=biased, loss_graph=RMSEDenseLossGraph)
rmse_dense_result = fit_and_eval(model_baseline, user_features, item_features, train_interactions, test_interactions,
                                 fit_kwargs)
print("RMSEDense: \t%s, \t%s" % rmse_dense_result)

model_baseline = TensorRec(n_components=n_components, biased=biased, loss_graph=SeparationLossGraph)
sep_result = fit_and_eval(model_baseline, user_features, item_features, train_interactions, test_interactions,
                          fit_kwargs)
print("Separation: \t%s, \t%s" % sep_result)

model_baseline = TensorRec(n_components=n_components, biased=biased, loss_graph=SeparationDenseLossGraph)
sep_dense_result = fit_and_eval(model_baseline, user_features, item_features, train_interactions, test_interactions,
                                fit_kwargs)
print("SepDense: \t%s, \t%s" % sep_dense_result)

model_wmrb = TensorRec(n_components=n_components, biased=biased, loss_graph=WMRBLossGraph)
wmrb_result = fit_and_eval(model_wmrb, user_features, item_features, train_interactions, test_interactions,
                           fit_kwargs)
print("WMRB: \t\t%s, \t%s" % wmrb_result)

model_wmrb = TensorRec(n_components=n_components, biased=biased, loss_graph=WMRBAlignmentLossGraph)
wmrb_alignment_result = fit_and_eval(model_wmrb, user_features, item_features, train_interactions, test_interactions,
                                     fit_kwargs)
print("WMRBAlignment: \t%s, \t%s" % wmrb_alignment_result)

print('--------------------------------------------------')
print("RMSE: \t\t%s, \t%s" % rmse_result)
print("RMSEDense: \t%s, \t%s" % rmse_dense_result)
print("Separation: \t%s, \t%s" % sep_result)
print("SepDense: \t%s, \t%s" % sep_dense_result)
print("WMRB: \t\t%s, \t%s" % wmrb_result)
print("WMRBAlignment: \t%s, \t%s" % wmrb_alignment_result)
