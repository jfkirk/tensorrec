import tensorflow as tf


def linear_representation_graph(tf_features, n_components, n_features, node_name_ending):
    # Rough approximation of http://ceur-ws.org/Vol-1448/paper4.pdf

    # Create variable nodes
    tf_linear_weights = tf.Variable(tf.random_normal([n_features, n_components], stddev=.5),
                                    name='linear_weights_%s' % node_name_ending)
    tf_repr = tf.sparse_tensor_dense_matmul(tf_features, tf_linear_weights)

    # Return repr layer and variables
    return tf_repr, [tf_linear_weights]


def relu_representation_graph(tf_features, n_components, n_features, node_name_ending):
    relu_size = 4 * n_components

    # Create variable nodes
    tf_relu_weights = tf.Variable(tf.random_normal([n_features, relu_size], stddev=.5),
                                  name='relu_weights_%s' % node_name_ending)
    tf_relu_biases = tf.Variable(tf.zeros([1, relu_size]),
                                 name='relu_biases_%s' % node_name_ending)
    tf_linear_weights = tf.Variable(tf.random_normal([relu_size, n_components], stddev=.5),
                                    name='linear_weights_%s' % node_name_ending)

    # Create ReLU layer
    tf_relu = tf.nn.relu(tf.add(tf.sparse_tensor_dense_matmul(tf_features, tf_relu_weights),
                                tf_relu_biases))
    tf_repr = tf.matmul(tf_relu, tf_linear_weights)

    # Return repr layer and variables
    return tf_repr, [tf_relu_weights, tf_linear_weights, tf_relu_biases]
