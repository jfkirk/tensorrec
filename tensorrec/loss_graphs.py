import abc
import tensorflow as tf


class AbstractLossGraph(object):
    __metaclass__ = abc.ABCMeta

    # If True, dense prediction results will be passed to the loss function
    is_dense = False

    # If True, randomly sampled predictions will be passed to the loss function
    is_sample_based = False
    # If True, and if is_sample_based is True, predictions will be sampled with replacement
    is_sampled_with_replacement = False

    @abc.abstractmethod
    def connect_loss_graph(self, tf_prediction_serial, tf_interactions_serial, tf_interactions, tf_n_users, tf_n_items,
                           tf_prediction, tf_rankings, tf_sample_predictions, tf_n_sampled_items):
        """
        This method is responsible for consuming a number of possible nodes from the graph and calculating loss from
        those nodes.

        The following parameters are always passed in
        :param tf_prediction_serial: tf.Tensor
        The recommendation scores as a Tensor of shape [n_samples, 1]
        :param tf_interactions_serial: tf.Tensor
        The sample interactions corresponding to tf_prediction_serial as a Tensor of shape [n_samples, 1]
        :param tf_interactions: tf.SparseTensor
        The sample interactions as a SparseTensor of shape [n_users, n_items]
        :param tf_n_users: tf.placeholder
        The number of users in tf_interactions
        :param tf_n_items: tf.placeholder
        The number of items in tf_interactions

        The following parameters are passed in if is_dense is True
        :param tf_prediction: tf.Tensor
        The recommendation scores as a Tensor of shape [n_users, n_items]
        :param tf_rankings: tf.Tensor
        The item ranks as a Tensor of shape [n_users, n_items]

        The following parameters are passed in if is_sample_based is True
        :param tf_sample_predictions: tf.Tensor
        The recommendation scores of a sample of items of shape [n_users, n_sampled_items]
        :param tf_n_sampled_items: tf.placeholder
        The number of items per user in tf_sample_predictions

        :return: tf.Tensor
        The loss value.
        """
        pass


class RMSELossGraph(AbstractLossGraph):
    """
    This loss function returns the root mean square error between the predictions and the true interactions.
    Interactions can be any positive or negative values, and this loss function is sensitive to magnitude.
    """
    def connect_loss_graph(self, tf_prediction_serial, tf_interactions_serial, **kwargs):
        return tf.sqrt(tf.reduce_mean(tf.square(tf_interactions_serial - tf_prediction_serial)))


class RMSEDenseLossGraph(AbstractLossGraph):
    """
    This loss function returns the root mean square error between the predictions and the true interactions, including
    all non-interacted values as 0s.
    Interactions can be any positive or negative values, and this loss function is sensitive to magnitude.
    """
    is_dense = True

    def connect_loss_graph(self, tf_interactions, tf_prediction, **kwargs):
        error = tf.sparse_add(tf_interactions, -1.0 * tf_prediction)
        return tf.sqrt(tf.reduce_mean(tf.square(error)))


class SeparationLossGraph(AbstractLossGraph):
    """
    This loss function models the explicit positive and negative interaction predictions as normal distributions and
    returns the probability of overlap between the two distributions.
    Interactions can be any positive or negative values, but this loss function ignores the magnitude of the
    interaction -- interactions are grouped in to {i <= 0} and {i > 0}.
    """
    def connect_loss_graph(self, tf_prediction_serial, tf_interactions_serial, **kwargs):

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


class SeparationDenseLossGraph(AbstractLossGraph):
    """
    This loss function models all positive and negative interaction predictions as normal distributions and
    returns the probability of overlap between the two distributions. This loss function includes non-interacted items
    as negative interactions.
    Interactions can be any positive or negative values, but this loss function ignores the magnitude of the
    interaction -- interactions are grouped in to {i <= 0} and {i > 0}.
    """
    is_dense = True

    def connect_loss_graph(self, tf_prediction, tf_interactions, **kwargs):

        interactions_shape = tf.shape(tf_interactions)
        int_serial_shape = tf.cast([interactions_shape[0] * interactions_shape[1]], tf.int32)
        tf_interactions_serial = tf.reshape(tf.sparse_tensor_to_dense(tf_interactions),
                                            shape=int_serial_shape)

        prediction_shape = tf.shape(tf_prediction)
        pred_serial_shape = tf.cast([prediction_shape[0] * prediction_shape[1]], tf.int32)
        tf_prediction_serial = tf.reshape(tf_prediction, shape=pred_serial_shape)

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


class WMRBLossGraph(AbstractLossGraph):
    """
    Approximation of http://ceur-ws.org/Vol-1905/recsys2017_poster3.pdf
    Interactions can be any positive values, but magnitude is ignored. Negative interactions are ignored.
    """
    is_sample_based = True

    def connect_loss_graph(self, tf_prediction_serial, tf_interactions, tf_sample_predictions, tf_n_items,
                           tf_n_sampled_items, **kwargs):

        return self.weighted_margin_rank_batch(tf_prediction_serial=tf_prediction_serial,
                                               tf_interactions=tf_interactions,
                                               tf_sample_predictions=tf_sample_predictions,
                                               tf_n_items=tf_n_items,
                                               tf_n_sampled_items=tf_n_sampled_items)

    def weighted_margin_rank_batch(self, tf_prediction_serial, tf_interactions, tf_sample_predictions, tf_n_items,
                                   tf_n_sampled_items):
        positive_interaction_mask = tf.greater(tf_interactions.values, 0.0)
        positive_interaction_indices = tf.boolean_mask(tf_interactions.indices,
                                                       positive_interaction_mask)

        # [ n_positive_interactions ]
        positive_predictions = tf.boolean_mask(tf_prediction_serial,
                                               positive_interaction_mask)

        n_items = tf.cast(tf_n_items, dtype=tf.float32)
        n_sampled_items = tf.cast(tf_n_sampled_items, dtype=tf.float32)

        # [ n_positive_interactions, n_sampled_items ]
        mapped_predictions_sample_per_interaction = tf.gather(params=tf_sample_predictions,
                                                              indices=tf.transpose(positive_interaction_indices)[0])

        # [ n_positive_interactions, n_sampled_items ]
        summation_term = tf.maximum(1.0
                                    - tf.expand_dims(positive_predictions, axis=1)
                                    + mapped_predictions_sample_per_interaction,
                                    0.0)

        # [ n_positive_interactions ]
        sampled_margin_rank = (n_items / n_sampled_items) * tf.reduce_sum(summation_term, axis=1)

        loss = tf.log(sampled_margin_rank + 1.0)
        return loss


class BalancedWMRBLossGraph(WMRBLossGraph):
    """
    This loss graph extends WMRB by making it sensitive to interaction magnitude and weighting the loss of each item by
    1 / sum(interactions) per item.
    Interactions can be any positive values. Negative interactions are ignored.
    """
    def weighted_margin_rank_batch(self, tf_prediction_serial, tf_interactions, tf_sample_predictions, tf_n_items,
                                   tf_n_sampled_items):
        positive_interaction_mask = tf.greater(tf_interactions.values, 0.0)
        positive_interaction_indices = tf.boolean_mask(tf_interactions.indices,
                                                       positive_interaction_mask)
        positive_interaction_values = tf.boolean_mask(tf_interactions.values,
                                                      positive_interaction_mask)

        positive_interactions = tf.SparseTensor(indices=positive_interaction_indices,
                                                values=positive_interaction_values,
                                                dense_shape=tf_interactions.dense_shape)
        listening_sum_per_item = tf.sparse_reduce_sum(positive_interactions, axis=0)
        gathered_sums = tf.gather(params=listening_sum_per_item,
                                  indices=tf.transpose(positive_interaction_indices)[1])

        # [ n_positive_interactions ]
        positive_predictions = tf.boolean_mask(tf_prediction_serial,
                                               positive_interaction_mask)

        n_items = tf.cast(tf_n_items, dtype=tf.float32)
        n_sampled_items = tf.cast(tf_n_sampled_items, dtype=tf.float32)

        # [ n_positive_interactions, n_sampled_items ]
        mapped_predictions_sample_per_interaction = tf.gather(params=tf_sample_predictions,
                                                              indices=tf.transpose(positive_interaction_indices)[0])

        # [ n_positive_interactions, n_sampled_items ]
        summation_term = tf.maximum(1.0
                                    - tf.expand_dims(positive_predictions, axis=1)
                                    + mapped_predictions_sample_per_interaction,
                                    0.0)

        # [ n_positive_interactions ]
        sampled_margin_rank = ((n_items / n_sampled_items)
                               * tf.reduce_sum(summation_term, axis=1)
                               * positive_interaction_values / gathered_sums)

        loss = tf.log(sampled_margin_rank + 1.0)
        return loss
