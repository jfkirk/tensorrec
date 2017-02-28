import itertools
import numpy as np
import random
import scipy.sparse as sp


def generate_sparse_input(interactions_matrix, user_features_matrix, item_features_matrix):

    # Coerce input data to malleable sparse types
    if not isinstance(interactions_matrix, sp.coo_matrix):
        interactions_matrix = sp.coo_matrix(interactions_matrix)
    if not isinstance(user_features_matrix, sp.csr_matrix):
        user_features_matrix = sp.csr_matrix(user_features_matrix)
    if not isinstance(item_features_matrix, sp.csr_matrix):
        item_features_matrix = sp.csr_matrix(item_features_matrix)

    batch_x_user = []
    batch_x_item = []
    batch_y = []
    batch_id_tuples = []

    for (user, item, y) in itertools.izip(interactions_matrix.row,
                                          interactions_matrix.col,
                                          interactions_matrix.data):
        batch_x_user.append(user_features_matrix[user])
        batch_x_item.append(item_features_matrix[item])
        batch_y.append(y)
        batch_id_tuples.append((user, item))

    return sp.coo_matrix(sp.vstack(batch_x_user)), sp.coo_matrix(sp.vstack(batch_x_item)), \
           np.asarray(batch_y), batch_id_tuples


def generate_dummy_data(num_users=15000, num_items=30000, interaction_density=.00045, pos_int_ratio=.5):


    n_user_features = int(num_users * 1.2)
    n_user_tags = num_users * 3
    n_item_features = int(num_items * 1.2)
    n_item_tags = num_items * 3
    n_interactions = (num_users * num_items) * interaction_density

    user_features = sp.lil_matrix((num_users, n_user_features))
    for i in range(num_users):
        user_features[i, i] = 1

    for i in range(n_user_tags):
        user_features[random.randrange(num_users), random.randrange(num_users, n_user_features)] = 1

    item_features = sp.lil_matrix((num_items, n_item_features))
    for i in range(num_items):
        item_features[i, i] = 1

    for i in range(n_item_tags):
        item_features[random.randrange(num_items), random.randrange(num_items, n_item_features)] = 1

    interactions = sp.lil_matrix((num_users, num_items))
    for i in range(int(n_interactions * pos_int_ratio)):
        interactions[random.randrange(num_users), random.randrange(num_items)] = 1

    for i in range(int(n_interactions * (1 - pos_int_ratio))):
        interactions[random.randrange(num_users), random.randrange(num_items)] = -1

    return interactions, user_features, item_features
