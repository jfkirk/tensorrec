import tensorflow as tf


def project_biases(tf_features, n_features):
    """
    Projects the biases from the feature space to calculate bias per actor
    :param tf_features:
    :param n_features:
    :return:
    """
    tf_feature_biases = tf.Variable(tf.zeros([n_features, 1]))

    # The reduce sum is to perform a rank reduction
    tf_projected_biases = tf.reduce_sum(
        tf.sparse_tensor_dense_matmul(tf_features, tf_feature_biases),
        axis=1
    )

    return tf_feature_biases, tf_projected_biases


def split_sparse_tensor_indices(tf_sparse_tensor, n_dimensions):
    """
    Separates the each dimension of a sparse tensor's indices into independent tensors
    :param tf_sparse_tensor:
    :param n_dimensions:
    :return:
    """
    tf_transposed_indices = tf.transpose(tf_sparse_tensor.indices)
    return (tf_transposed_indices[i] for i in range(n_dimensions))


def bias_prediction_dense(tf_prediction, tf_projected_user_biases, tf_projected_item_biases):
    """
    Broadcasts user and item biases across their respective axes of the dense predictions
    :param tf_prediction:
    :param tf_projected_user_biases:
    :param tf_projected_item_biases:
    :return:
    """
    return tf_prediction + tf.expand_dims(tf_projected_user_biases, 1) + tf.expand_dims(tf_projected_item_biases, 0)


def bias_prediction_serial(tf_prediction_serial, tf_projected_user_biases, tf_projected_item_biases, tf_x_user,
                           tf_x_item):
    """
    Calculates the bias per user/item pair and adds it to the serial predictions
    :param tf_prediction_serial:
    :param tf_projected_user_biases:
    :param tf_projected_item_biases:
    :param tf_x_user:
    :param tf_x_item:
    :return:
    """
    gathered_user_biases = tf.gather(tf_projected_user_biases, tf_x_user)
    gathered_item_biases = tf.gather(tf_projected_item_biases, tf_x_item)
    return tf_prediction_serial + gathered_user_biases + gathered_item_biases


def gather_sampled_item_predictions(tf_prediction, tf_sampled_item_indices):
    """
    Gathers the predictions for the given sampled items.
    :param tf_prediction:
    :param tf_sampled_item_indices:
    :return:
    """
    prediction_shape = tf.shape(tf_prediction)
    flattened_prediction = tf.reshape(tf_prediction, shape=[prediction_shape[0] * prediction_shape[1]])

    indices_shape = tf.shape(tf_sampled_item_indices)
    flattened_indices = tf.reshape(tf_sampled_item_indices, shape=[indices_shape[0] * indices_shape[1]])

    gathered_predictions = tf.gather(params=flattened_prediction, indices=flattened_indices)
    reshaped_gathered_predictions = tf.reshape(gathered_predictions, shape=indices_shape)
    return reshaped_gathered_predictions


def rank_predictions(tf_prediction):
    """
    Double-sortation serves as a ranking process
    The +1 is so the top-ranked has a non-zero rank
    :param tf_prediction:
    :return:
    """
    tf_prediction_item_size = tf.shape(tf_prediction)[1]
    tf_indices_of_ranks = tf.nn.top_k(tf_prediction, k=tf_prediction_item_size)[1]
    return tf.nn.top_k(-tf_indices_of_ranks, k=tf_prediction_item_size)[1] + 1
