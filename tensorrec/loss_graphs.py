import tensorflow as tf


def rmse_loss(tf_prediction_serial, tf_interactions_serial, **kwargs):
    """
    This loss function returns the root mean square error between the predictions and the true interactions.
    :param tf_prediction_serial:
    :param tf_interactions_serial:
    :return:
    """
    return tf.sqrt(tf.reduce_mean(tf.square(tf_interactions_serial - tf_prediction_serial)))


def separation_loss(tf_prediction_serial, tf_interactions_serial, **kwargs):
    """
    This loss function models the explicit positive and negative interaction predictions as normal distributions and
    returns the probability of overlap between the two distributions.
    :param tf_prediction_serial:
    :param tf_interactions_serial:
    :return:
    """

    tf_positive_mask = tf.greater(tf_interactions_serial, 0.0)
    tf_negative_mask = tf.less_equal(tf_interactions_serial, 0.0)

    tf_positive_predictions = tf.boolean_mask(tf_prediction_serial, tf_positive_mask)
    tf_negative_predictions = tf.boolean_mask(tf_prediction_serial, tf_negative_mask)

    tf_pos_mean, tf_pos_var = tf.nn.moments(tf_positive_predictions, axes=[0])
    tf_neg_mean, tf_neg_var = tf.nn.moments(tf_negative_predictions, axes=[0])

    tf_overlap_distribution = tf.contrib.distributions.Normal(loc=(tf_neg_mean - tf_pos_mean),
                                                              scale=tf.sqrt(tf_neg_var + tf_pos_var))

    loss = 1.0 - tf_overlap_distribution.cdf(0.0)
    return loss


def warp_loss(tf_prediction, tf_y, **kwargs):
    # TODO JK: implement WARP loss

    tf_positive_mask = tf.greater(tf_y, 0.0)
    tf_negative_mask = tf.less_equal(tf_y, 0.0)

    tf_positive_predictions = tf.boolean_mask(tf_prediction, tf_positive_mask) # noqa
    tf_negative_predictions = tf.boolean_mask(tf_prediction, tf_negative_mask) # noqa
