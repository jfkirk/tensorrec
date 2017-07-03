import numpy as np

import tensorflow as tf

from .tensorrec import TensorRec


def precision_at_k(model, session, test_interactions, k=10, user_features=None, item_features=None,
                   preserve_rows=False):
    """
    Modified from LightFM.
    :param model:
    :param session:
    :param test_interactions:
    :param k:
    :param user_features:
    :param item_features:
    :param preserve_rows:
    :return:
    """
    ranks = model.predict_rank(session, test_interactions,
                               user_features=user_features,
                               item_features=item_features)

    ranks.data = np.less(ranks.data, k, ranks.data)

    precision = np.squeeze(np.array(ranks.sum(axis=1))) / k

    if not preserve_rows:
        precision = precision[test_interactions.getnnz(axis=1) > 0]

    return precision


def recall_at_k(model, session, test_interactions, k=10, user_features=None, item_features=None, preserve_rows=False):
    """
    Modified from LightFM.
    :param model:
    :param session:
    :param test_interactions:
    :param k:
    :param user_features:
    :param item_features:
    :param preserve_rows:
    :return:
    """
    ranks = model.predict_rank(session, test_interactions,
                               user_features=user_features,
                               item_features=item_features)

    ranks.data = np.less(ranks.data, k, ranks.data)

    retrieved = np.squeeze(test_interactions.getnnz(axis=1))
    hit = np.squeeze(np.array(ranks.sum(axis=1)))

    if not preserve_rows:
        hit = hit[test_interactions.getnnz(axis=1) > 0]
        retrieved = retrieved[test_interactions.getnnz(axis=1) > 0]

    return hit / retrieved


def fit_and_eval(model, user_features, item_features, train_interactions, test_interactions, fit_kwargs):

    session = tf.Session()

    model.fit(session=session, user_features=user_features, item_features=item_features, interactions=train_interactions, **fit_kwargs)
    p_at_k = precision_at_k(model, session, test_interactions,
                            user_features=user_features,
                            item_features=item_features,
                            k=100)
    r_at_k = recall_at_k(model, session, test_interactions,
                         user_features=user_features,
                         item_features=item_features,
                         k=100)

    return np.mean(r_at_k), np.mean(p_at_k)


def grid_check_model_on_dataset(train_interactions, test_interactions, user_features, item_features):

    results = []
    for n_components in [2, 4, 8, 16, 32, 64, 128, 256]:
        for epochs in [2, 4, 8, 16, 32, 64, 128, 256]:
            model = TensorRec(n_components=n_components)
            scores = fit_and_eval(model, user_features, item_features, train_interactions, test_interactions,
                                  fit_kwargs={'epochs': epochs})
            results.append((n_components, scores))
            print (n_components, epochs, scores)
