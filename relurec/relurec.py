import itertools
import numpy as np
from scipy import sparse as sp
import tensorflow as tf

from .loss import build_separation_loss

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
    tf_repr = tf.matmul(tf_relu_with_dropout, tf_tanh_weights)

    # Return TanH layer and variables
    return tf_repr, [tf_relu_weights, tf_tanh_weights, tf_relu_biases]


class ReLURec(object):

    def __init__(self, n_components=100, num_threads=4, session=None):

        self.n_components = n_components
        self.num_threads = num_threads

        self.tf_user_representation = None
        self.tf_item_representation = None
        self.tf_affinity = None
        self.tf_projected_user_biases = None
        self.tf_projected_item_biases = None
        self.tf_prediction_sparse = None
        self.tf_prediction_dense = None

        # TF feed placeholders
        self.tf_n_users = None
        self.tf_n_items = None
        self.tf_nn_dropout_keep_proba = None
        self.tf_user_features = None
        self.tf_item_features = None
        self.tf_user_feature_indices = None
        self.tf_item_feature_indices = None
        self.tf_user_feature_values = None
        self.tf_item_feature_values = None
        self.tf_y = None
        self.tf_x_user = None
        self.tf_x_item = None

        # For l2 normalization
        self.tf_weights = []

        self.session = session or tf.Session()

    def create_feed_dict(self, interactions_matrix, user_features_matrix, item_features_matrix, dropout_keep_proba):

        # Check that input data is of a sparse type
        if not sp.issparse(interactions_matrix):
            raise Exception('Interactions must be a scipy sparse matrix')
        if not sp.issparse(user_features_matrix):
            raise Exception('User features must be a scipy sparse matrix')
        if not sp.issparse(item_features_matrix):
            raise Exception('Item features must be a scipy sparse matrix')

        # Coerce input data to zippable sparse types
        if not isinstance(interactions_matrix, sp.coo_matrix):
            interactions_matrix = sp.coo_matrix(interactions_matrix)
        if not isinstance(user_features_matrix, sp.coo_matrix):
            user_features_matrix = sp.coo_matrix(user_features_matrix)
        if not isinstance(item_features_matrix, sp.coo_matrix):
            item_features_matrix = sp.coo_matrix(item_features_matrix)

        feed_dict = {self.tf_n_users: user_features_matrix.shape[0],
                     self.tf_n_items: item_features_matrix.shape[0],
                     self.tf_user_feature_indices: zip(user_features_matrix.row, user_features_matrix.col),
                     self.tf_user_feature_values: user_features_matrix.data,
                     self.tf_item_feature_indices: zip(item_features_matrix.row, item_features_matrix.col),
                     self.tf_item_feature_values: item_features_matrix.data,
                     self.tf_x_user: interactions_matrix.row,
                     self.tf_x_item: interactions_matrix.col,
                     self.tf_y: interactions_matrix.data,
                     self.tf_nn_dropout_keep_proba: dropout_keep_proba}

        return feed_dict

    def build_tf_graph(self, n_user_features, n_item_features):

        # Initialize placeholder values for inputs
        self.tf_n_users = tf.placeholder('int64')
        self.tf_n_items = tf.placeholder('int64')
        self.tf_nn_dropout_keep_proba = tf.placeholder('float')
        self.tf_user_feature_indices = tf.placeholder('int64', [None, 2])
        self.tf_user_feature_values = tf.placeholder('float', None)
        self.tf_item_feature_indices = tf.placeholder('int64', [None, 2])
        self.tf_item_feature_values = tf.placeholder('float', None)
        self.tf_y = tf.placeholder('float', [None], name='y')
        self.tf_x_user = tf.placeholder('int64', None, name='x_user')
        self.tf_x_item = tf.placeholder('int64', None, name='x_item')

        # Construct the features as sparse matrices
        self.tf_user_features = tf.SparseTensor(self.tf_user_feature_indices, self.tf_user_feature_values,
                                         [self.tf_n_users, n_user_features])
        self.tf_item_features = tf.SparseTensor(self.tf_item_feature_indices, self.tf_item_feature_values,
                                         [self.tf_n_items, n_item_features])

        # Build the representations
        self.tf_user_representation, user_weights = \
            _build_representation_graph(tf_features=self.tf_user_features,
                                        tf_nn_dropout_keep_proba=self.tf_nn_dropout_keep_proba,
                                        n_components=self.n_components,
                                        n_features=n_user_features,
                                        node_name_ending='user')
        self.tf_item_representation, item_weights = \
            _build_representation_graph(tf_features=self.tf_item_features,
                                        tf_nn_dropout_keep_proba=self.tf_nn_dropout_keep_proba,
                                        n_components=self.n_components,
                                        n_features=n_item_features,
                                        node_name_ending='item')

        gathered_user_reprs = tf.gather(self.tf_user_representation, self.tf_x_user)
        gathered_item_reprs = tf.gather(self.tf_item_representation, self.tf_x_item)

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

        gathered_user_biases = tf.gather(self.tf_projected_user_biases, self.tf_x_user)
        gathered_item_biases = tf.gather(self.tf_projected_item_biases, self.tf_x_item)

        # Prediction = user_repr * item_repr + user_bias + item_bias
        # The reduce sum is to perform a rank reduction
        self.tf_prediction_sparse = (tf.reduce_sum(tf.multiply(gathered_user_reprs,
                                                               gathered_item_reprs), axis=1)
                                     + gathered_user_biases + gathered_item_biases)

        self.tf_prediction_dense = (tf.reduce_sum(tf.matmul(self.tf_user_representation,
                                                            self.tf_item_representation,
                                                            transpose_b=True), axis=1)
                                    + self.tf_projected_user_biases + self.tf_projected_item_biases)

        self.tf_weights = []
        self.tf_weights.extend(user_weights)
        self.tf_weights.extend(item_weights)
        self.tf_weights.append(tf_user_feature_biases)
        self.tf_weights.append(tf_item_feature_biases)

    def fit(self, interactions, user_features, item_features, epochs=100, learning_rate=0.01, alpha=0.001,
            end_on_loss_increase=False, verbose=True, out_sample_interactions=None):

        # Numbers of features are learned at fit time from the shape of these two matrices and cannot be changed without
        # refitting
        self.build_tf_graph(n_user_features=user_features.shape[1], n_item_features=item_features.shape[1])

        if verbose:
            print 'Processing interaction and feature data'

        feed_dict = self.create_feed_dict(interactions, user_features, item_features, dropout_keep_proba=.5)

        if out_sample_interactions is not None:
            os_feed_dict = self.create_feed_dict(out_sample_interactions, user_features, item_features,
                                                 dropout_keep_proba=1.0)

        basic_loss = build_separation_loss(tf_prediction=self.tf_prediction_sparse,
                                           tf_y=self.tf_y)
        weight_reg_loss = sum(tf.nn.l2_loss(weights) for weights in self.tf_weights)

        tf_loss = basic_loss + alpha * weight_reg_loss
        tf_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)

        if verbose:
            print 'Beginning fitting'

        self.session.run(tf.global_variables_initializer())
        avg_loss = basic_loss.eval(session=self.session, feed_dict=feed_dict)
        smooth_delta_loss = 0.0
        for epoch in range(epochs):

            self.session.run(tf_optimizer, feed_dict=feed_dict)

            # TODO JK: These if statements are ugly as hell
            if verbose or end_on_loss_increase:
                lst_loss = avg_loss
                avg_loss = basic_loss.eval(session=self.session, feed_dict=feed_dict)
                smooth_delta_loss = smooth_delta_loss * .95 + (lst_loss - avg_loss) * .05

            if verbose:
                avg_pred = np.mean(self.tf_prediction_sparse.eval(session=self.session, feed_dict=feed_dict))
                l2_loss = (alpha * weight_reg_loss).eval(session=self.session, feed_dict=feed_dict)
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

        feed_dict = self.create_feed_dict(dummy_interactions, user_features, item_features, dropout_keep_proba=1.0)

        predictions = self.tf_prediction_sparse.eval(session=self.session, feed_dict=feed_dict)

        return predictions

    def predict_rank(self, test_interactions, user_features, item_features, train_interactions=None):

        feed_dict = self.create_feed_dict(test_interactions, user_features, item_features, dropout_keep_proba=1.0)

        predictions = self.tf_prediction_dense.eval(session=self.session, feed_dict=feed_dict)
