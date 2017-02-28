import numpy as np
from scipy import sparse as sp
import tensorflow as tf

from .util import generate_sparse_input

__all__ = ['ReLURec']


def build_representation_graph(tf_x, tf_nn_dropout_keep_proba, n_components, n_features, node_name_ending):
    relu_size = 4 * n_components

    tf_relu_weights = tf.Variable(tf.random_normal([n_features, relu_size]),
                                  name='relu_weights_%s' % node_name_ending)
    tf_relu_biases = tf.Variable(tf.zeros([1, relu_size]),
                                 name='relu_biases_%s' % node_name_ending)
    tf_tanh_weights = tf.Variable(tf.random_normal([relu_size, n_components]),
                                  name='tanh_weights_%s' % node_name_ending)

    tf_relu = tf.nn.dropout(tf.nn.relu(tf.add(tf.sparse_tensor_dense_matmul(tf_x, tf_relu_weights), tf_relu_biases)),
                            keep_prob=tf_nn_dropout_keep_proba)

    return tf.tanh(tf.matmul(tf_relu, tf_tanh_weights)), tf_relu_weights, tf_relu_biases, tf_tanh_weights


class ReLURec(object):

    def __init__(self, n_components=100, num_threads=4, session=None):

        self.n_components = n_components
        self.num_threads = num_threads

        self.tf_user_representation = None
        self.tf_item_representation = None
        self.tf_affinity = None
        self.tf_projected_user_biases = None
        self.tf_projected_item_biases = None
        self.tf_prediction = None

        # TF variable placeholders
        self.tf_user_feature_biases = None
        self.tf_item_feature_biases = None
        self.tf_relu_weights_user = None
        self.tf_relu_weights_item = None
        self.tf_relu_biases_user = None
        self.tf_relu_biases_item = None
        self.tf_tanh_weights_user = None
        self.tf_tanh_weights_item = None

        # TF feed placeholders
        self.tf_nn_dropout_keep_proba = None
        self.tf_x_user = None
        self.tf_x_item = None
        self.tf_x_user_indices = None
        self.tf_x_item_indices = None
        self.tf_x_user_values = None
        self.tf_x_item_values = None
        self.tf_n_examples = None
        self.tf_y = None

        self.session = session or tf.Session()

    def build_tf_graph(self, n_user_features, n_item_features):

        # Initialize placeholder values for inputs
        self.tf_nn_dropout_keep_proba = tf.placeholder("float")
        self.tf_n_examples = tf.placeholder("int64")
        self.tf_x_user_indices = tf.placeholder("int64", [None, 2])
        self.tf_x_user_values = tf.placeholder("float", None)
        self.tf_x_item_indices = tf.placeholder("int64", [None, 2])
        self.tf_x_item_values = tf.placeholder("float", None)
        self.tf_y = tf.placeholder("float", None, name='y')

        # Construct the features as sparse matrices
        self.tf_x_user = tf.SparseTensor(self.tf_x_user_indices, self.tf_x_user_values,
                                         [self.tf_n_examples, n_user_features])
        self.tf_x_item = tf.SparseTensor(self.tf_x_item_indices, self.tf_x_item_values,
                                         [self.tf_n_examples, n_item_features])

        # Fire the TanH layer
        self.tf_user_representation, self.tf_relu_weights_user, self.tf_relu_biases_user, self.tf_tanh_weights_user = \
            build_representation_graph(tf_x=self.tf_x_user,
                                       tf_nn_dropout_keep_proba=self.tf_nn_dropout_keep_proba,
                                       n_components=self.n_components,
                                       n_features=n_user_features,
                                       node_name_ending='user')
        self.tf_item_representation, self.tf_relu_weights_item, self.tf_relu_biases_item, self.tf_tanh_weights_item = \
            build_representation_graph(tf_x=self.tf_x_item,
                                       tf_nn_dropout_keep_proba=self.tf_nn_dropout_keep_proba,
                                       n_components=self.n_components,
                                       n_features=n_item_features,
                                       node_name_ending='item')

        # Calculate the user and item biases
        self.tf_user_feature_biases = tf.Variable(tf.zeros([n_user_features, 1]))
        self.tf_item_feature_biases = tf.Variable(tf.zeros([n_item_features, 1]))
        self.tf_projected_user_biases = tf.reduce_sum(tf.sparse_tensor_dense_matmul(self.tf_x_user, self.tf_user_feature_biases), axis=1)
        self.tf_projected_item_biases = tf.reduce_sum(tf.sparse_tensor_dense_matmul(self.tf_x_item, self.tf_item_feature_biases), axis=1)

        # Prediction = user_repr * item_repr + user_bias + item_bias
        # The reduce sum is to perform a rank reduction
        self.tf_prediction = (tf.reduce_sum(tf.multiply(self.tf_user_representation,
                                                        self.tf_item_representation), axis=1)
                              + self.tf_projected_user_biases + self.tf_projected_item_biases)

    def fit(self, interactions_matrix, user_features, item_features, epochs=100, learning_rate=0.01, alpha=0.00001,
            beta=0.00001, end_on_loss_increase=False, verbose=True):

        self.build_tf_graph(n_user_features=user_features.shape[1], n_item_features=item_features.shape[1])

        basic_loss = tf.sqrt(tf.reduce_mean(tf.square(self.tf_y - self.tf_prediction)))
        weight_reg_loss = (tf.nn.l2_loss(self.tf_relu_weights_user) + tf.nn.l2_loss(self.tf_relu_weights_item)
                           + tf.nn.l2_loss(self.tf_tanh_weights_user) + tf.nn.l2_loss(self.tf_tanh_weights_item))
        bias_reg_loss = (tf.nn.l2_loss(self.tf_user_feature_biases) + tf.nn.l2_loss(self.tf_item_feature_biases)
                         + tf.nn.l2_loss(self.tf_relu_biases_user) + tf.nn.l2_loss(self.tf_relu_biases_item))

        tf_loss = basic_loss + alpha * weight_reg_loss + beta * bias_reg_loss
        tf_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)

        if verbose:
            print 'Processing interaction and feature data'

        sparse_x_user, sparse_x_item, sparse_y, _ = generate_sparse_input(interactions_matrix,
                                                                          user_features,
                                                                          item_features)
        feed_dict = {self.tf_n_examples: len(sparse_y),
                     self.tf_x_user_indices: zip(sparse_x_user.row, sparse_x_user.col),
                     self.tf_x_user_values: sparse_x_user.data,
                     self.tf_x_item_indices: zip(sparse_x_item.row, sparse_x_item.col),
                     self.tf_x_item_values: sparse_x_item.data,
                     self.tf_y: sparse_y,
                     self.tf_nn_dropout_keep_proba: .9}

        if verbose:
            print 'Beginning fitting'

        self.session.run(tf.global_variables_initializer())
        avg_loss = basic_loss.eval(session=self.session, feed_dict=feed_dict)
        smooth_delta_loss = 0.0
        for epoch in range(epochs):

            self.session.run(tf_optimizer, feed_dict=feed_dict)

            if verbose or end_on_loss_increase:
                lst_loss = avg_loss
                avg_loss = basic_loss.eval(session=self.session, feed_dict=feed_dict)
                smooth_delta_loss = smooth_delta_loss * .95 + (lst_loss - avg_loss) * .05

            if verbose:
                avg_pred = np.mean(self.tf_prediction.eval(session=self.session, feed_dict=feed_dict))
                l2_loss = (alpha * weight_reg_loss + beta * bias_reg_loss).eval(session=self.session,
                                                                                feed_dict=feed_dict)
                print 'EPOCH %s loss = %s, l2 = %s, smooth_delta_loss = %s, mean_pred = %s' % (epoch, avg_loss,
                                                                                               l2_loss,
                                                                                               smooth_delta_loss,
                                                                                               avg_pred)

            # Break when no longer improving
            if end_on_loss_increase and (smooth_delta_loss < 0.0) and (epoch > 10):
                break

    def predict(self, user_ids, item_ids, user_features, item_features):

        user_ids = np.asarray(user_ids, dtype=np.int32)
        item_ids = np.asarray(item_ids, dtype=np.int32)

        dummy_interactions = sp.dok_matrix((max(user_ids) + 1, max(item_ids) + 1))
        for user in user_ids:
            for item in item_ids:
                dummy_interactions[user, item] = 1

        sparse_x_user, sparse_x_item, sparse_y, sparse_indices = generate_sparse_input(dummy_interactions,
                                                                                       user_features,
                                                                                       item_features)

        feed_dict = {self.tf_n_examples: len(sparse_y),
                     self.tf_x_user_indices: zip(sparse_x_user.row, sparse_x_user.col),
                     self.tf_x_user_values: sparse_x_user.data,
                     self.tf_x_item_indices: zip(sparse_x_item.row, sparse_x_item.col),
                     self.tf_x_item_values: sparse_x_item.data,
                     self.tf_nn_dropout_keep_proba: 1.0}
        predictions = self.tf_prediction.eval(session=self.session, feed_dict=feed_dict)

        return zip(sparse_indices, predictions)
