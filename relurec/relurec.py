import numpy as np
from scipy import sparse as sp
import tensorflow as tf

from .loss import build_separation_loss
from .util import generate_sparse_input

__all__ = ['ReLURec']


def _build_representation_graph(tf_features, tf_nn_dropout_keep_proba, n_components, n_features, node_name_ending):
    relu_size = 4 * n_components

    # Create variable nodes
    tf_relu_weights = tf.Variable(tf.random_normal([n_features, relu_size]),
                                  name='relu_weights_%s' % node_name_ending)
    tf_relu_biases = tf.Variable(tf.zeros([1, relu_size]),
                                 name='relu_biases_%s' % node_name_ending)
    tf_tanh_weights = tf.Variable(tf.random_normal([relu_size, n_components]),
                                  name='tanh_weights_%s' % node_name_ending)

    # Create ReLU layer and TanH layer
    tf_relu = tf.nn.relu(tf.add(tf.sparse_tensor_dense_matmul(tf_features, tf_relu_weights),
                                tf_relu_biases))
    tf_relu_with_dropout = tf.nn.dropout(tf_relu, keep_prob=tf_nn_dropout_keep_proba)
    tf_tanh = tf.tanh(tf.matmul(tf_relu_with_dropout, tf_tanh_weights))

    # Return TanH layer and variables
    return tf_tanh, [tf_relu_weights, tf_tanh_weights], [tf_relu_biases]


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

        # TF feed placeholders
        self.tf_nn_dropout_keep_proba = None
        self.tf_user_features = None
        self.tf_item_features = None
        self.tf_user_feature_indices = None
        self.tf_item_feature_indices = None
        self.tf_user_feature_values = None
        self.tf_item_feature_values = None
        self.tf_n_examples = None
        self.tf_y = None

        # For l2 normalization
        self.tf_weights = []
        self.tf_biases = []

        self.session = session or tf.Session()

    def build_tf_graph(self, n_user_features, n_item_features):

        # Initialize placeholder values for inputs
        self.tf_nn_dropout_keep_proba = tf.placeholder('float')
        self.tf_n_examples = tf.placeholder('int64')
        self.tf_user_feature_indices = tf.placeholder('int64', [None, 2])
        self.tf_user_feature_values = tf.placeholder('float', None)
        self.tf_item_feature_indices = tf.placeholder('int64', [None, 2])
        self.tf_item_feature_values = tf.placeholder('float', None)
        self.tf_y = tf.placeholder('float', [None], name='y')

        # Construct the features as sparse matrices
        self.tf_user_features = tf.SparseTensor(self.tf_user_feature_indices, self.tf_user_feature_values,
                                         [self.tf_n_examples, n_user_features])
        self.tf_item_features = tf.SparseTensor(self.tf_item_feature_indices, self.tf_item_feature_values,
                                         [self.tf_n_examples, n_item_features])

        # Build the representations
        self.tf_user_representation, user_weights, user_biases = \
            _build_representation_graph(tf_features=self.tf_user_features,
                                        tf_nn_dropout_keep_proba=self.tf_nn_dropout_keep_proba,
                                        n_components=self.n_components,
                                        n_features=n_user_features,
                                        node_name_ending='user')
        self.tf_item_representation, item_weights, item_biases = \
            _build_representation_graph(tf_features=self.tf_item_features,
                                        tf_nn_dropout_keep_proba=self.tf_nn_dropout_keep_proba,
                                        n_components=self.n_components,
                                        n_features=n_item_features,
                                        node_name_ending='item')

        # Calculate the user and item biases
        tf_user_feature_biases = tf.Variable(tf.zeros([n_user_features, 1]))
        tf_item_feature_biases = tf.Variable(tf.zeros([n_item_features, 1]))

        self.tf_projected_user_biases = tf.reduce_sum(
            tf.sparse_tensor_dense_matmul(self.tf_user_features, tf_user_feature_biases),
            axis=1
        )
        self.tf_projected_item_biases = tf.reduce_sum(
            tf.sparse_tensor_dense_matmul(self.tf_item_features, tf_item_feature_biases),
            axis=1
        )

        # Prediction = user_repr * item_repr + user_bias + item_bias
        # The reduce sum is to perform a rank reduction
        self.tf_prediction = (tf.reduce_sum(tf.multiply(self.tf_user_representation,
                                                        self.tf_item_representation), axis=1)
                              + self.tf_projected_user_biases + self.tf_projected_item_biases)

        self.tf_weights = []
        self.tf_weights.extend(user_weights)
        self.tf_weights.extend(item_weights)

        self.tf_biases = []
        self.tf_biases.extend(user_biases)
        self.tf_biases.extend(item_biases)
        self.tf_biases.append(tf_user_feature_biases)
        self.tf_biases.append(tf_item_feature_biases)

    def fit(self, interactions_matrix, user_features, item_features, epochs=100, learning_rate=0.01, alpha=0.00001,
            beta=0.00001, end_on_loss_increase=False, verbose=True, out_sample_interactions=None):

        self.build_tf_graph(n_user_features=user_features.shape[1], n_item_features=item_features.shape[1])

        if verbose:
            print 'Processing interaction and feature data'

        coo_user_features, coo_item_features, y_values, _ = \
            generate_sparse_input(interactions_matrix, user_features, item_features)

        feed_dict = {self.tf_n_examples: len(y_values),
                     self.tf_user_feature_indices: zip(coo_user_features.row, coo_user_features.col),
                     self.tf_user_feature_values: coo_user_features.data,
                     self.tf_item_feature_indices: zip(coo_item_features.row, coo_item_features.col),
                     self.tf_item_feature_values: coo_item_features.data,
                     self.tf_y: y_values,
                     self.tf_nn_dropout_keep_proba: .9}

        if out_sample_interactions is not None:
            os_coo_user_features, os_coo_item_features, os_y_values, _ = \
                generate_sparse_input(out_sample_interactions, user_features, item_features)

            os_feed_dict = {self.tf_n_examples: len(os_y_values),
                            self.tf_user_feature_indices: zip(os_coo_user_features.row, os_coo_user_features.col),
                            self.tf_user_feature_values: os_coo_user_features.data,
                            self.tf_item_feature_indices: zip(os_coo_item_features.row, os_coo_item_features.col),
                            self.tf_item_feature_values: os_coo_item_features.data,
                            self.tf_y: os_y_values,
                            self.tf_nn_dropout_keep_proba: 1.0}

        basic_loss = build_separation_loss(tf_prediction=self.tf_prediction,
                                           tf_y=self.tf_y)
        weight_reg_loss = sum(tf.nn.l2_loss(weights) for weights in self.tf_weights)
        bias_reg_loss = sum(tf.nn.l2_loss(biases) for biases in self.tf_biases)

        tf_loss = basic_loss + alpha * weight_reg_loss + beta * bias_reg_loss
        tf_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)

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

            if (out_sample_interactions is not None) and verbose:
                os_loss = basic_loss.eval(session=self.session, feed_dict=os_feed_dict)
                print 'Out-Sample loss = %s' % os_loss

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

        coo_user_features, coo_item_features, y_values, id_tuples = \
            generate_sparse_input(dummy_interactions, user_features, item_features)

        feed_dict = {self.tf_n_examples: len(y_values),
                     self.tf_user_feature_indices: zip(coo_user_features.row, coo_user_features.col),
                     self.tf_user_feature_values: coo_user_features.data,
                     self.tf_item_feature_indices: zip(coo_item_features.row, coo_item_features.col),
                     self.tf_item_feature_values: coo_item_features.data,
                     self.tf_nn_dropout_keep_proba: 1}

        print self.tf_user_representation.eval(session=self.session, feed_dict=feed_dict)

        predictions = self.tf_prediction.eval(session=self.session, feed_dict=feed_dict)

        return zip(id_tuples, predictions)
