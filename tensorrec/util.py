import math
import numpy as np
import random
import scipy.sparse as sp
import tensorflow as tf


def sample_items(n_items, n_users, n_sampled_items, replace):
    items_per_user = [np.random.choice(a=n_items, size=n_sampled_items, replace=replace)
                      for _ in range(n_users)]

    sample_indices = []
    for user, users_items in enumerate(items_per_user):
        for item in users_items:
            sample_indices.append((user, item))

    return sample_indices


def calculate_batched_alpha(num_batches, alpha):
    if num_batches < 1:
        raise ValueError('num_batches must be >=1, num_batches={}'.format(num_batches))
    elif num_batches > 1:
        batched_alpha = alpha / (math.e * math.log(num_batches))
    else:
        batched_alpha = alpha
    return batched_alpha


def generate_dummy_data(num_users=15000, num_items=30000, interaction_density=.00045, num_user_features=200,
                        num_item_features=200, n_features_per_user=20, n_features_per_item=20,  pos_int_ratio=.5):

    if pos_int_ratio <= 0.0:
        raise Exception("pos_int_ratio must be > 0")

    print("Generating positive interactions")
    interactions = sp.rand(num_users, num_items, density=interaction_density * pos_int_ratio)
    if pos_int_ratio < 1.0:
        print("Generating negative interactions")
        interactions += -1 * sp.rand(num_users, num_items, density=interaction_density * (1 - pos_int_ratio))

    print("Generating user features")
    user_features = sp.rand(num_users, num_user_features, density=float(n_features_per_user) / num_user_features)

    print("Generating item features")
    item_features = sp.rand(num_items, num_item_features, density=float(n_features_per_item) / num_item_features)

    return interactions, user_features, item_features


def generate_dummy_data_with_indicator(num_users=15000, num_items=30000, interaction_density=.00045, pos_int_ratio=.5):

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


def append_to_string_at_point(string, value, point):
    for _ in range(0, (point - len(string))):
        string += " "
    string += "{}".format(value)
    return string


def simple_tf_print(tensor, places=100):
    return tf.Print(tensor, [tensor, tf.shape(tensor)], summarize=places)
