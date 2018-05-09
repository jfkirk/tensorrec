import numpy as np
import scipy.sparse as sp

from .tensorrec import TensorRec


def precision_at_k(predicted_ranks, test_interactions, k=10, preserve_rows=False):
    """
    Modified from LightFM.
    :param predicted_ranks: Numpy matrix
    The results of model.predict_rank()
    :param test_interactions: scipy.sparse matrix
    Test interactions matrix of shape [n_users, n_items]
    :param k: int
    The rank at which to stop evaluating precision.
    :param preserve_rows: bool
    If True, a value of 0 will be returned for every user without test interactions.
    If False, only users with test interactions will be returned.
    :return: np.array
    """
    positive_test_interactions = test_interactions > 0
    ranks = sp.csr_matrix(predicted_ranks * positive_test_interactions.A)
    ranks.data = np.less(ranks.data, (k + 1), ranks.data)

    precision = np.squeeze(np.array(ranks.sum(axis=1))).astype(float) / k

    if not preserve_rows:
        precision = precision[positive_test_interactions.getnnz(axis=1) > 0]

    return precision


def recall_at_k(predicted_ranks, test_interactions, k=10, preserve_rows=False):
    """
    Modified from LightFM.
    :param predicted_ranks: Numpy matrix
    The results of model.predict_rank()
    :param test_interactions: scipy.sparse matrix
    Test interactions matrix of shape [n_users, n_items]
    :param k: int
    The rank at which to stop evaluating recall.
    :param preserve_rows: bool
    If True, a value of 0 will be returned for every user without test interactions.
    If False, only users with test interactions will be returned.
    :return: np.array
    """
    positive_test_interactions = test_interactions > 0
    ranks = sp.csr_matrix(predicted_ranks * positive_test_interactions.A)
    ranks.data = np.less(ranks.data, (k + 1), ranks.data)

    retrieved = np.squeeze(positive_test_interactions.getnnz(axis=1))
    hit = np.squeeze(np.array(ranks.sum(axis=1)))

    if not preserve_rows:
        hit = hit[positive_test_interactions.getnnz(axis=1) > 0]
        retrieved = retrieved[positive_test_interactions.getnnz(axis=1) > 0]

    return hit.astype(float) / retrieved.astype(float)


def _setup_ndcg(predicted_ranks, test_interactions, k=10):

    pos_inter = test_interactions > 0
    ror = sp.csr_matrix(predicted_ranks * pos_inter.A)

    relevance = sp.csr_matrix(test_interactions.A * pos_inter.A)

    k_mask = np.less(ror.data, k + 1)
    ror_at_k = np.maximum(np.multiply(ror.data, k_mask), 1)

    return relevance, k_mask, ror, ror_at_k


def _idcg(hits, k=10):
    sorted_hits = hits[np.argsort(-hits)][:min(len(hits), k)]
    idgc = np.sum((2**sorted_hits-1)/np.log2(np.arange(len(sorted_hits)) + 2))
    return idgc


def _dcg(relevance, k_mask, ror_at_k, ror):
    numer = (2**np.multiply(relevance.data, k_mask)) - 1
    denom = np.log2(ror_at_k + 1)
    ror.data = numer/denom  # ranks at 1
    dcg = ror.sum(axis=1).flatten()

    return dcg


def ndcg_at_k(predicted_ranks, test_interactions, k=10, preserve_rows=False):
    """
    Calculate Normalized Discounted Cumulative Gain @K.
    :param predicted_ranks: Numpy matrix
    The results of model.predict_rank()
    :param test_interactions: scipy.sparse matrix
    Test interactions matrix of shape [n_users, n_items]
    :param k: int
    The rank at which to stop evaluating NDCG.
    :param preserve_rows: bool
    If True, a value of 0 will be returned for every user without test interactions.
    If False, only users with test interactions will be returned.
    :return: np.array
    """

    relevance, k_mask, ranks_of_relevant, ror_at_k = _setup_ndcg(predicted_ranks,
                                                                 test_interactions,
                                                                 k)

    dcg = np.asarray(_dcg(relevance, k_mask, ror_at_k, ranks_of_relevant))[0]
    idcg = np.apply_along_axis(_idcg, 1, relevance.A)

    ndcg = dcg/idcg

    if not preserve_rows:
        positive_test_interactions = test_interactions > 0
        ndcg = ndcg[positive_test_interactions.getnnz(axis=1) > 0]

    return ndcg


def f1_score_at_k(predicted_ranks, test_interactions, k=10, preserve_rows=False):
    """
    :param model: TensorRec
    A trained TensorRec model.
    :param test_interactions: scipy.sparse matrix
    Test interactions matrix of shape [n_users, n_items]
    :param user_features: scipy.sparse matrix
    User features matrix of shape [n_users, n_user_features]
    :param item_features: scipy.sparse matrix
    Item features matrix of shape [n_items, n_item_features]
    :param k: int
    The rank at which to stop evaluating recall.
    :param preserve_rows: bool
    If True, a value of 0 will be returned for every user without test interactions.
    If False, only users with test interactions will be returned.
    :return: np.array
    """
    p_at_k = precision_at_k(predicted_ranks=predicted_ranks,
                            test_interactions=test_interactions,
                            k=k, preserve_rows=preserve_rows)
    r_at_k = recall_at_k(predicted_ranks=predicted_ranks,
                         test_interactions=test_interactions,
                         k=k, preserve_rows=preserve_rows)

    mean_p = np.mean(p_at_k)
    mean_r = np.mean(r_at_k)

    f1_score = (2.0 * mean_p * mean_r) / (mean_p + mean_r)
    return f1_score


def fit_and_eval(model, user_features, item_features, train_interactions, test_interactions, fit_kwargs, recall_k=30,
                 precision_k=5, ndcg_k=30):

    model.fit(user_features=user_features, item_features=item_features,
              interactions=train_interactions, **fit_kwargs)
    predicted_ranks = model.predict_rank(user_features=user_features, item_features=item_features)

    p_at_k = precision_at_k(predicted_ranks, test_interactions, k=precision_k)
    r_at_k = recall_at_k(predicted_ranks, test_interactions, k=recall_k)
    n_at_k = ndcg_at_k(predicted_ranks, test_interactions, k=ndcg_k)

    p_at_k_insample = precision_at_k(predicted_ranks, train_interactions, k=precision_k)
    r_at_k_insample = recall_at_k(predicted_ranks, train_interactions, k=recall_k)
    n_at_k_insample = ndcg_at_k(predicted_ranks, train_interactions, k=ndcg_k)

    return (np.mean(r_at_k), np.mean(p_at_k), np.mean(n_at_k), np.mean(r_at_k_insample), np.mean(p_at_k_insample),
            np.mean(n_at_k_insample))


def grid_check_model_on_dataset(train_interactions, test_interactions, user_features, item_features):

    results = []
    for n_components in [2, 4, 8, 16, 32, 64, 128, 256]:
        for epochs in [2, 4, 8, 16, 32, 64, 128, 256]:
            model = TensorRec(n_components=n_components)
            scores = fit_and_eval(model, user_features, item_features, train_interactions, test_interactions,
                                  fit_kwargs={'epochs': epochs})
            results.append((n_components, scores))
            print(n_components, epochs, scores)


def eval_random_ranks_on_dataset(interactions, recall_k=30, precision_k=5, ndcg_k=30):

    n_users, n_items = interactions.shape

    random_guesses = np.array([np.random.choice(a=n_items, size=n_items, replace=False) + 1 for _ in range(n_users)])

    p_at_k = precision_at_k(random_guesses, interactions, k=precision_k)
    r_at_k = recall_at_k(random_guesses, interactions, k=recall_k)
    n_at_k = ndcg_at_k(random_guesses, interactions, k=ndcg_k)

    return np.mean(r_at_k), np.mean(p_at_k), np.mean(n_at_k)
