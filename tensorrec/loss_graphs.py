import abc
import tensorflow as tf


class AbstractLossGraph(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, is_dense=False):
        self.is_dense = is_dense

    @abc.abstractmethod
    def loss_graph(self, tf_prediction_serial, tf_interactions_serial, tf_prediction, tf_interactions, tf_rankings,
                   tf_alignment):
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
        :return: tf.Tensor
        The loss value.
        """
        pass


class RMSELoss(AbstractLossGraph):
    """
    This loss function returns the root mean square error between the predictions and the true interactions.
    Interactions can be any positive or negative values, and this loss function is sensitive to magnitude.
    """
    def loss_graph(self, tf_prediction_serial, tf_interactions_serial, **kwargs):
        return tf.sqrt(tf.reduce_mean(tf.square(tf_interactions_serial - tf_prediction_serial)))


class SeparationLoss(AbstractLossGraph):
    """
    This loss function models the explicit positive and negative interaction predictions as normal distributions and
    returns the probability of overlap between the two distributions.
    Interactions can be any positive or negative values, but this loss function ignored the magnitude of the
    interaction -- interactions are grouped in to {i < 0} and {i > 0}.
    """
    def loss_graph(self, tf_prediction_serial, tf_interactions_serial, **kwargs):

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


class WMRBLoss(AbstractLossGraph):
    """
    Approximation of http://ceur-ws.org/Vol-1905/recsys2017_poster3.pdf
    Interactions can be any positive values, but magnitude is ignored. Negative interactions are also ignored.
    """
    def __init__(self):
        super(WMRBLoss, self).__init__(is_dense=True)

    def loss_graph(self, tf_interactions, tf_prediction, **kwargs):

        positive_interaction_mask = tf.greater(tf_interactions.values, 0.0)
        positive_interaction_indices = tf.boolean_mask(tf_interactions.indices,
                                                       positive_interaction_mask)
        positive_predictions = tf.gather_nd(tf_prediction, indices=positive_interaction_indices)

        n_items = tf.cast(tf.shape(tf_prediction)[1], dtype=tf.float32)
        predictions_sum_per_user = tf.reduce_sum(tf_prediction, axis=1)
        mapped_predictions_sum_per_user = tf.gather(params=predictions_sum_per_user,
                                                    indices=tf.transpose(positive_interaction_indices)[0])

        # TODO smart irrelevant item indicator -- using n_items is an approximation for sparse interactions
        irrelevant_item_indicator = n_items # noqa

        sampled_margin_rank = (n_items - (n_items * positive_predictions)
                               + mapped_predictions_sum_per_user + irrelevant_item_indicator)

        # JKirk - I am leaving out the log term due to experimental results
        # loss = tf.log(sampled_margin_rank + 1.0)
        return sampled_margin_rank


class WMRBAlignmentLoss(WMRBLoss):
    """
    Approximation of http://ceur-ws.org/Vol-1905/recsys2017_poster3.pdf
    Ranks items based on alignment, in place of prediction.
    Interactions can be any positive values, but magnitude is ignored. Negative interactions are also ignored.
    """
    def __init__(self):
        super(WMRBAlignmentLoss, self).__init__()

    def loss_graph(self, tf_interactions, tf_alignment, **kwargs):
        return super(WMRBAlignmentLoss, self).loss_graph(tf_interactions=tf_interactions, tf_prediction=tf_alignment)
