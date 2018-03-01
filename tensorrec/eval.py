import numpy as np
import scipy.sparse as sp

from .tensorrec import TensorRec


def precision_at_k(model, test_interactions, k=10, user_features=None, item_features=None,
                   preserve_rows=False):
    """
    Modified from LightFM.
    :param model:
    :param test_interactions:
    :param k:
    :param user_features:
    :param item_features:
    :param preserve_rows:
    :return:
    """

    predicted_ranks = model.predict_rank(user_features=user_features, item_features=item_features)

    positive_test_interactions = test_interactions > 0
    ranks = sp.csr_matrix(predicted_ranks * positive_test_interactions.A)
    ranks.data = np.less(ranks.data, (k + 1), ranks.data)

    precision = np.squeeze(np.array(ranks.sum(axis=1))) / k

    if not preserve_rows:
        precision = precision[positive_test_interactions.getnnz(axis=1) > 0]

    return precision


def recall_at_k(model, test_interactions, k=10, user_features=None, item_features=None, preserve_rows=False):
    """
    Modified from LightFM.
    :param model:
    :param test_interactions:
    :param k:
    :param user_features:
    :param item_features:
    :param preserve_rows:
    :return:
    """

    predicted_ranks = model.predict_rank(user_features=user_features, item_features=item_features)

    positive_test_interactions = test_interactions > 0
    ranks = sp.csr_matrix(predicted_ranks * positive_test_interactions.A)
    ranks.data = np.less(ranks.data, (k + 1), ranks.data)

    retrieved = np.squeeze(positive_test_interactions.getnnz(axis=1))
    hit = np.squeeze(np.array(ranks.sum(axis=1)))

    if not preserve_rows:
        hit = hit[positive_test_interactions.getnnz(axis=1) > 0]
        retrieved = retrieved[positive_test_interactions.getnnz(axis=1) > 0]

    return hit / retrieved


def _idcg(hits, k=10, ctype="binary"):

    if ctype == "binary":
        arg = min(hits, k+1)
        idgc = np.sum(1/np.log2(np.arange(arg) + 2))  # arange index from 0
    elif ctype == "scalar":
        sorted = hits[np.argsort(-hits)][:min(len(hits), k)]
        idgc = np.sum(sorted/np.log2(np.arange(len(sorted)) + 2))
    else:
        raise ValueError("Invalid IDCG calculation type")

    return idgc


def normalized_discounted_cumulative_gain(model, test_interactions, k=10,
                                          user_features=None,
                                          item_features=None,
                                          preserve_rows=False):
    """
    Calculate Normalize Discounted Cumulative Gain @K.
    :param model: prediction model
    :param test_interactions: test interactions
    :param k:
    :param user_features:
    :param item_features:
    :param preserve_rows: If true, return NDCG per row. If false, return mean NDCG
    :return:
    """

    predicted_ranks = model.predict_rank(user_features=user_features,
                                         item_features=item_features)

    positive_test_interactions = test_interactions > 0
    ranks_of_relevant = sp.csr_matrix(predicted_ranks *
                                      positive_test_interactions.A)

    k_mask = np.less(ranks_of_relevant.data, k + 1)
    ror_at_k = np.maximum(np.multiply(ranks_of_relevant.data, k_mask), 1)

    relevance = sp.csr_matrix(
        test_interactions.A *
        positive_test_interactions.A
    )

    relevance_at_k = (2**np.multiply(relevance.data, k_mask)) - 1
    ranks_of_relevant.data = relevance_at_k/np.log2(ror_at_k + 1)  # ranks at 1

    dcg = ranks_of_relevant.sum(axis=1).flatten()

    if np.max(test_interactions.A) == 1:
        # If the data is binary, we can save the ideal ranking sort
        idcg = np.apply_along_axis(_idcg,
                                   0,
                                   positive_test_interactions.sum(axis=1).flatten()[0],
                                   ctype="binary"
                                   )
    else:
        idcg = np.apply_along_axis(_idcg,
                                   0,
                                   relevance,
                                   ctype="scalar"
                                   )

    ndcg = dcg/idcg

    if preserve_rows:
        return ndcg.flatten()
    else:
        return np.nanmean(ndcg.flatten())


def f1_score_at_k(model, test_interactions, k=10, user_features=None, item_features=None, preserve_rows=False):
    # TODO: Refactor to calculate more quickly
    p_at_k = precision_at_k(model=model,
                            test_interactions=test_interactions,
                            k=k, user_features=user_features,
                            item_features=item_features,
                            preserve_rows=preserve_rows)
    r_at_k = recall_at_k(model=model,
                         test_interactions=test_interactions,
                         k=k, user_features=user_features,
                         item_features=item_features,
                         preserve_rows=preserve_rows)

    mean_p = np.mean(p_at_k)
    mean_r = np.mean(r_at_k)

    f1_score = (2.0 * mean_p * mean_r) / (mean_p + mean_r)
    return f1_score


def fit_and_eval(model, user_features, item_features, train_interactions, test_interactions, fit_kwargs):

    model.fit(user_features=user_features, item_features=item_features,
              interactions=train_interactions, **fit_kwargs)
    p_at_k = precision_at_k(model, test_interactions,
                            user_features=user_features,
                            item_features=item_features,
                            k=100)
    r_at_k = recall_at_k(model, test_interactions,
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
            print(n_components, epochs, scores)
