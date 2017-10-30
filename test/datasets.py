import numpy as np
import scipy.sparse as sp

from lightfm import datasets


def get_movielens_100k(min_positive_score=4):
    movielens_100k_dict = datasets.fetch_movielens(indicator_features=True, genre_features=True)

    def flip_ratings(ratings_matrix):
        ratings_matrix.data = np.array([1 if rating >= min_positive_score else -1 for rating in ratings_matrix.data])
        return ratings_matrix

    test_interactions = flip_ratings(movielens_100k_dict['test'])
    train_interactions = flip_ratings(movielens_100k_dict['train'])

    # Create indicator features for all users
    num_users = train_interactions.shape[0]
    user_features = sp.identity(num_users)

    return train_interactions, test_interactions, user_features, movielens_100k_dict['item_features']
