import tensorflow as tf


def build_separation_loss(tf_prediction, tf_y):

    tf_positive_mask = tf.greater(tf_y, 0.0)
    tf_negative_mask = tf.less_equal(tf_y, 0.0)

    tf_positive_predictions = tf.boolean_mask(tf_prediction, tf_positive_mask)
    tf_negative_predictions = tf.boolean_mask(tf_prediction, tf_negative_mask)

    tf_pos_mean, tf_pos_var = tf.nn.moments(tf_positive_predictions, axes=[0])
    tf_neg_mean, tf_neg_var = tf.nn.moments(tf_negative_predictions, axes=[0])

    tf_overlap_gaussian = tf.contrib.distributions.Normal(mu=(tf_neg_mean - tf_pos_mean),
                                                          sigma=tf.sqrt(tf_neg_var + tf_pos_var))

    loss = 1.0 - tf_overlap_gaussian.cdf(0.0)
    return loss


def build_rmse_loss(tf_prediction, tf_y):
    return tf.sqrt(tf.reduce_mean(tf.square(tf_y - tf_prediction)))
