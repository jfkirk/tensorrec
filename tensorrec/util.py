import math
import numpy as np
import random
import scipy.sparse as sp
import six
import tensorflow as tf


def sample_items(n_items, n_users, n_sampled_items, replace):
    items_per_user = [np.random.choice(a=n_items, size=n_sampled_items, replace=replace)
                      for _ in range(n_users)]

    sample_indices = []
    for user, users_items in enumerate(items_per_user):
        for item in users_items:
            sample_indices.append((user, item))

    return np.array(sample_indices)


def calculate_batched_alpha(num_batches, alpha):
    if num_batches < 1:
        raise ValueError('num_batches must be >=1, num_batches={}'.format(num_batches))
    elif num_batches > 1:
        batched_alpha = alpha / (math.e * math.log(num_batches))
    else:
        batched_alpha = alpha
    return batched_alpha


def datasets_from_raw_input(raw_input, contains_counter):

    if isinstance(raw_input, tf.data.Dataset):
        return [raw_input]

    if sp.issparse(raw_input):
        return handle_sparse_matrix_input(input_sparse_matrix=raw_input, contains_counter=contains_counter)

    if isinstance(raw_input, list) or isinstance(raw_input, set):

        if all([sp.issparse(input_val) for input_val in raw_input]):
            return handle_sparse_matrix_list_input(input_sparse_matrix_list=raw_input,
                                                   contains_counter=contains_counter)

        if all([isinstance(input_val, tf.data.Dataset) for input_val in raw_input]):
            return raw_input

    raise ValueError('Input must be a scipy sparse matrix, an iterable of scipy sprase matrices, or a TensorFlow '
                     'Dataset')


def handle_sparse_matrix_input(input_sparse_matrix, contains_counter):
    return handle_sparse_matrix_list_input(input_sparse_matrix_list=[input_sparse_matrix],
                                           contains_counter=contains_counter)


def handle_sparse_matrix_list_input(input_sparse_matrix_list, contains_counter):
    all_tensor_slices = [tensor_slices_from_sparse_matrix(input_matrix, contains_counter)
                         for input_matrix in input_sparse_matrix_list]
    return [tf.data.Dataset.from_tensor_slices(tensor_slices) for tensor_slices in all_tensor_slices]


def tensor_slices_from_sparse_matrix(sparse_matrix, contains_counter):

    if not isinstance(sparse_matrix, sp.coo_matrix):
        sparse_matrix = sp.coo_matrix(sparse_matrix)

    # "Actors" is used to signify "users or items" -- unused for interactions
    n_actors = np.array([sparse_matrix.shape[0]], dtype=np.int64)
    feature_indices = np.array([[pair for pair in six.moves.zip(sparse_matrix.row, sparse_matrix.col)]],
                               dtype=np.int64)
    feature_values = np.array([sparse_matrix.data], dtype=np.float32)

    if contains_counter:
        tensor_slices = (feature_indices, feature_values, n_actors)
    else:
        tensor_slices = (feature_indices, feature_values)

    return tensor_slices


def generate_dummy_data(num_users=15000, num_items=30000, interaction_density=.00045, num_user_features=200,
                        num_item_features=200, n_features_per_user=20, n_features_per_item=20,  pos_int_ratio=.5,
                        return_datasets=False):

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

    if return_datasets:
        interactions = handle_sparse_matrix_input(interactions, contains_counter=False)[0]
        user_features = handle_sparse_matrix_input(user_features, contains_counter=True)[0]
        item_features = handle_sparse_matrix_input(item_features, contains_counter=True)[0]

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
