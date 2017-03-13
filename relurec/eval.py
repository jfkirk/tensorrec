import numpy as np

from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score

from .util import generate_movielens_data


def fit_and_eval(model, user_features, item_features, train_interactions, test_interactions, fit_kwargs):

    model.fit(user_features=user_features, item_features=item_features, interactions=train_interactions, **fit_kwargs)
    p_at_k = precision_at_k(model, test_interactions,
                            train_interactions=train_interactions,
                            user_features=user_features,
                            item_features=item_features,
                            k=100)
    roc_auc = auc_score(model, test_interactions,
                        train_interactions=train_interactions,
                        user_features=user_features,
                        item_features=item_features)

    return np.mean(roc_auc), np.mean(p_at_k)


def check_model_on_movielens():
    train_interactions, test_interactions, user_features, item_features = generate_movielens_data(4.0)

    results = []
    for no_components in [2, 4, 8, 16, 32, 64, 128, 256]:
        for epochs in [2, 4, 8, 16, 32, 64, 128, 256]:
            model = LightFM(no_components=no_components, loss='warp')
            scores = fit_and_eval(model, user_features, item_features, train_interactions, test_interactions,
                                  fit_kwargs={'epochs': epochs})
            results.append((no_components, scores))
            print (no_components, epochs, scores)
