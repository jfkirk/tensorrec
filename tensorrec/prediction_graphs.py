import abc
import tensorflow as tf

from .recommendation_graphs import relative_cosine


class AbstractPredictionGraph(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def connect_dense_prediction_graph(self, tf_user_representation, tf_item_representation):
        """
        This method is responsible for consuming user and item representations and calculating prediction scores for all
        possible user-item pairs based on these representations.
        :param tf_user_representation: tf.Tensor
        The user representations as a Tensor of shape [n_users, n_components]
        :param tf_item_representation: tf.Tensor
        The item representations as a Tensor of shape [n_items, n_components]
        :return: tf.Tensor
        The predictions as a Tensor of shape [n_users, n_items]
        """
        pass

    @abc.abstractmethod
    def connect_serial_prediction_graph(self, tf_user_representation, tf_item_representation, tf_x_user, tf_x_item):
        """
        This method is responsible for consuming user and item representations and indices and calculating prediction
        scores for particular user-item pairs.
        :param tf_user_representation: tf.Tensor
        The user representations as a Tensor of shape [n_users, n_components]
        :param tf_item_representation: tf.Tensor
        The item representations as a Tensor of shape [n_items, n_components]
        :param tf_x_user: tf.Tensor
        The users for whom to predict as a Tensor of shape [n_interactions]
        :param tf_x_item: tf.Tensor
        The items for which to predict as a Tensor of shape [n_interactions]
        :return: tf.Tensor
        The predictions as a Tensor of shape [n_interactions]
        """
        pass


class DotProductPredictionGraph(AbstractPredictionGraph):
    """
    This prediction function calculates the prediction as the dot product between the user and item representations.
    Prediction = user_repr * item_repr
    """

    def connect_dense_prediction_graph(self, tf_user_representation, tf_item_representation):
        return tf.matmul(tf_user_representation, tf_item_representation, transpose_b=True)

    def connect_serial_prediction_graph(self, tf_user_representation, tf_item_representation, tf_x_user, tf_x_item):
        gathered_user_reprs = tf.gather(tf_user_representation, tf_x_user)
        gathered_item_reprs = tf.gather(tf_item_representation, tf_x_item)
        return tf.reduce_sum(tf.multiply(gathered_user_reprs, gathered_item_reprs), axis=1)


class CosineSimilarityPredictionGraph(AbstractPredictionGraph):
    """
    This prediction function calculates the prediction as the cosine between the user and item representations.
    Prediction = cos(user_repr, item_repr)
    """

    def connect_dense_prediction_graph(self, tf_user_representation, tf_item_representation):
        return relative_cosine(tf_tensor_1=tf_user_representation, tf_tensor_2=tf_item_representation)

    def connect_serial_prediction_graph(self, tf_user_representation, tf_item_representation, tf_x_user, tf_x_item):
        normalized_users = tf.nn.l2_normalize(tf_user_representation, 1)
        normalized_items = tf.nn.l2_normalize(tf_item_representation, 1)
        gathered_user_reprs = tf.gather(normalized_users, tf_x_user)
        gathered_item_reprs = tf.gather(normalized_items, tf_x_item)
        return tf.reduce_sum(tf.multiply(gathered_user_reprs, gathered_item_reprs), axis=1)


class EuclidianSimilarityPredictionGraph(AbstractPredictionGraph):
    """
    This prediction function calculates the prediction as the negative euclidian distance between the user and
    item representations.
    Prediction = -1 * sqrt(sum((user_repr - item_repr)^2))
    """

    epsilon = 1e-16

    def connect_dense_prediction_graph(self, tf_user_representation, tf_item_representation):

        # [ n_users, 1 ]
        r_user = tf.reduce_sum(tf_user_representation ** 2, 1, keep_dims=True)

        # [ n_items, 1 ]
        r_item = tf.reduce_sum(tf_item_representation ** 2, 1, keep_dims=True)

        # [ n_users, n_items ]
        distance = (r_user
                    - 2.0 * tf.matmul(tf_user_representation, tf_item_representation, transpose_b=True)
                    + tf.transpose(r_item))

        # For numeric stability
        distance = tf.maximum(distance, self.epsilon)

        return -1.0 * tf.sqrt(distance)

    def connect_serial_prediction_graph(self, tf_user_representation, tf_item_representation, tf_x_user, tf_x_item):

        # [ n_interactions, n_components ]
        gathered_user_reprs = tf.gather(tf_user_representation, tf_x_user)
        gathered_item_reprs = tf.gather(tf_item_representation, tf_x_item)

        # [ n_interactions, n_components ]
        delta = tf.pow(gathered_user_reprs - gathered_item_reprs, 2)

        # [ n_interactions, 1 ]
        distance = tf.reduce_sum(delta, axis=1)

        # For numeric stability
        distance = tf.maximum(distance, self.epsilon)

        return -1.0 * tf.sqrt(distance)
