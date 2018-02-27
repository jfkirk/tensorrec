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
    def loss_graph(self, tf_prediction_serial, tf_interactions_serial, tf_prediction, tf_interactions, tf_rankings,
                   tf_alignment, tf_sample_predictions, tf_sample_alignments):
        """
        This method is responsible for consuming a number of possible nodes from the graph and calculating loss from
        those nodes.
        :param tf_prediction_serial: tf.Tensor
        The recommendation scores as a Tensor of shape [n_samples, 1]
        :param tf_interactions_serial: tf.Tensor
        The sample interactions corresponding to tf_prediction_serial as a Tensor of shape [n_samples, 1]
        :param tf_prediction: tf.Tensor
        The recommendation scores as a Tensor of shape [n_users, n_items]
        :param tf_interactions: tf.SparseTensor
        The sample interactions as a SparseTensor of shape [n_users, n_items]
        :param tf_rankings: tf.Tensor
        The item ranks as a Tensor of shape [n_users, n_items]
        :param tf_alignment: tf.Tensor
        The item alignments as a Tensor of shape [n_users, n_items]
        :param tf_sample_predictions: tf.Tensor
        The recommendation scores of a sample of items of shape [n_users, n_sampled_items]
        :param tf_sample_alignments: tf.Tensor
        The alignments of a sample of items of shape [n_users, n_sampled_items]
        :return: tf.Tensor
        The loss value.
        """
        pass


class RMSELossGraph(AbstractLossGraph):
    """
    This loss function returns the root mean square error between the predictions and the true interactions.
    Interactions can be any positive or negative values, and this loss function is sensitive to magnitude.
    """
    def loss_graph(self, tf_prediction_serial, tf_interactions_serial, **kwargs):
        return tf.sqrt(tf.reduce_mean(tf.square(tf_interactions_serial - tf_prediction_serial)))


class RMSEDenseLossGraph(AbstractLossGraph):
    """
    This loss function returns the root mean square error between the predictions and the true interactions, including
    all non-interacted values as 0s.
    Interactions can be any positive or negative values, and this loss function is sensitive to magnitude.
    """
    is_dense = True

    def loss_graph(self, tf_interactions, tf_prediction, **kwargs):
        error = tf.sparse_add(tf_interactions, -1.0 * tf_prediction)
        return tf.sqrt(tf.reduce_mean(tf.square(error)))


class WMRBLossGraph(AbstractLossGraph):
    """
    Approximation of http://ceur-ws.org/Vol-1905/recsys2017_poster3.pdf
    Interactions can be any positive values, but magnitude is ignored. Negative interactions are also ignored.
    """
    is_dense = True
    is_sample_based = True

    def loss_graph(self, tf_prediction, tf_interactions, tf_sample_predictions, **kwargs):

        # WMRB expects bounded predictions
        tanh_prediction = tf.nn.sigmoid(tf_prediction)
        tanh_sample_prediction = tf.nn.sigmoid(tf_sample_predictions)

        return self.weighted_margin_rank_batch(tf_prediction=tanh_prediction,
                                               tf_interactions=tf_interactions,
                                               tf_sample_predictions=tanh_sample_prediction)

    @classmethod
    def weighted_margin_rank_batch(cls, tf_prediction, tf_interactions, tf_sample_predictions):
        positive_interaction_mask = tf.greater(tf_interactions.values, 0.0)
        positive_interaction_indices = tf.boolean_mask(tf_interactions.indices,
                                                       positive_interaction_mask)

        # [ n_positive_interactions ]
        positive_predictions = tf.gather_nd(tf_prediction, indices=positive_interaction_indices)

        n_items = tf.cast(tf.shape(tf_prediction)[1], dtype=tf.float32)
        n_sampled_items = tf.cast(tf.shape(tf_sample_predictions)[1], dtype=tf.float32)

        # [ n_positive_interactions, n_sampled_items ]
        mapped_predictions_sample_per_interaction = tf.gather(params=tf_sample_predictions,
                                                              indices=tf.transpose(positive_interaction_indices)[0])

        # [ n_positive_interactions, n_sampled_items ]
        summation_term = tf.abs(1.0
                                - tf.expand_dims(positive_predictions, axis=1)
                                + mapped_predictions_sample_per_interaction)

        # [ n_positive_interactions, 1 ]
        sampled_margin_rank = (n_items / n_sampled_items) * tf.reduce_sum(summation_term, axis=0)

        loss = tf.log(sampled_margin_rank + 1.0)
        return loss


class WMRBAlignmentLossGraph(WMRBLossGraph):
    """
    Approximation of http://ceur-ws.org/Vol-1905/recsys2017_poster3.pdf
    Ranks items based on alignment, in place of prediction.
    Interactions can be any positive values, but magnitude is ignored. Negative interactions are also ignored.
    """
    def loss_graph(self, tf_alignment, tf_interactions, tf_sample_alignments, **kwargs):
        return self.weighted_margin_rank_batch(tf_prediction=tf_alignment,
                                               tf_interactions=tf_interactions,
                                               tf_sample_predictions=tf_sample_alignments)
