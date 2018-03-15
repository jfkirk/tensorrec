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


def densify_sampled_item_predictions(tf_sample_predictions_serial, tf_n_sampled_items, tf_n_users):
    """
    Turns the serial predictions of the sample items in to a dense matrix of shape [ n_users, n_sampled_items ]
    :param tf_sample_predictions_serial:
    :param tf_n_sampled_items:
    :param tf_n_users:
    :return:
    """
    densified_shape = tf.cast(tf.stack([tf_n_users, tf_n_sampled_items]), tf.int32)
    densified_predictions = tf.reshape(tf_sample_predictions_serial, shape=densified_shape)
    return densified_predictions


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


def collapse_mixture_of_tastes(tastes_predictions):
    """
    Collapses a list of prediction nodes in to a single prediction node.
    :param tastes_predictions:
    :return:
    """
    stacked_tastes = tf.stack(tastes_predictions)
    max_prediction = tf.reduce_max(stacked_tastes, axis=0)
    return max_prediction
