import abc
import tensorflow as tf


class AbstractRepresentationGraph(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def connect_representation_graph(self, tf_features, n_components, n_features, node_name_ending):
        pass


class LinearRepresentationGraph(AbstractRepresentationGraph):
    """
    Calculates the representation by passing the features through a linear embedding.
    Rough approximation of http://ceur-ws.org/Vol-1448/paper4.pdf
    """

    def connect_representation_graph(self, tf_features, n_components, n_features, node_name_ending):
        # Create variable nodes
        tf_linear_weights = tf.Variable(tf.random_normal([n_features, n_components], stddev=.5),
                                        name='linear_weights_{}'.format(node_name_ending))
        tf_repr = tf.sparse_tensor_dense_matmul(tf_features, tf_linear_weights)

        # Return repr layer and variables
        return tf_repr, [tf_linear_weights]


class ReLURepresentationGraph(AbstractRepresentationGraph):
    """
    Calculates the repesentations by passing the features through a single-layer ReLU neural network.
    :param relu_size: int or None
    The number of nodes in the ReLU layer. If None, the layer will be of size 4*n_components.
    """

    def __init__(self, relu_size=None):
        self.relu_size = relu_size

    def connect_representation_graph(self, tf_features, n_components, n_features, node_name_ending):

        # Infer ReLU layer size if necessary
        if self.relu_size is None:
            relu_size = 4 * n_components
        else:
            relu_size = self.relu_size

        # Create variable nodes
        tf_relu_weights = tf.Variable(tf.random_normal([n_features, relu_size], stddev=.5),
                                      name='relu_weights_{}'.format(node_name_ending))
        tf_relu_biases = tf.Variable(tf.zeros([1, relu_size]),
                                     name='relu_biases_{}'.format(node_name_ending))
        tf_linear_weights = tf.Variable(tf.random_normal([relu_size, n_components], stddev=.5),
                                        name='linear_weights_{}'.format(node_name_ending))

        # Create ReLU layer
        tf_relu = tf.nn.relu(tf.add(tf.sparse_tensor_dense_matmul(tf_features, tf_relu_weights),
                                    tf_relu_biases))
        tf_repr = tf.matmul(tf_relu, tf_linear_weights)

        # Return repr layer and variables
        return tf_repr, [tf_relu_weights, tf_linear_weights, tf_relu_biases]
