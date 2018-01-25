import numpy as np
from scipy import sparse as sp
import six
import tensorflow as tf

from .loss_graphs import rmse_loss
from .representation_graphs import linear_representation_graph
from .session_management import get_session


class TensorRec(object):

    def __init__(self, n_components=100,
                 user_repr_graph=linear_representation_graph,
                 item_repr_graph=linear_representation_graph,
                 loss_graph=rmse_loss,
                 biased=True):
        """
        A TensorRec recommendation model.
        :param n_components: Integer
        The dimension of a single output of the representation function. Must be >= 1.
        :param user_repr_graph: Method
        A method which creates TensorFlow nodes to calculate the user representation.
        See tensorrec.representation_graphs for examples.
        :param item_repr_graph: Method
        A method which creates TensorFlow nodes to calculate the item representation.
        See tensorrec.representation_graphs for examples.
        :param loss_graph: Method
        A method which creates TensorFlow nodes to calculate the loss function.
        See tensorrec.loss_graphs for examples.
        :param biased: Boolean
        If True, a bias value will be calculated for every user feature and item feature.
        """

        # Arg-check
        if (n_components is None) or (user_repr_graph is None) or (item_repr_graph is None) or \
                (loss_graph is None):
            raise ValueError("All arguments to TensorRec() must be non-None")
        if n_components < 1:
            raise ValueError("n_components must be >= 1")

        self.n_components = n_components
        self.user_repr_graph_factory = user_repr_graph
        self.item_repr_graph_factory = item_repr_graph
        self.loss_graph_factory = loss_graph
        self.biased = biased

        self.tf_user_representation = None
        self.tf_item_representation = None
        self.tf_prediction_serial = None
        self.tf_prediction = None
        self.tf_rankings = None

        # Bias nodes
        if self.biased:
            self.tf_user_feature_biases = None
            self.tf_item_feature_biases = None
            self.tf_projected_user_biases = None
            self.tf_projected_item_biases = None

        # Training nodes
        self.tf_basic_loss = None
        self.tf_weight_reg_loss = None
        self.tf_loss = None
        self.tf_optimizer = None

        # TF feed placeholders
        self.tf_n_users = None
        self.tf_n_items = None
        self.tf_user_features = None
        self.tf_item_features = None
        self.tf_user_feature_indices = None
        self.tf_user_feature_values = None
        self.tf_item_feature_indices = None
        self.tf_item_feature_values = None
        self.tf_interaction_indices = None
        self.tf_interaction_values = None
        self.tf_learning_rate = None
        self.tf_alpha = None

        # For weight normalization
        self.tf_weights = []

    def _create_feed_dict(self, interactions_matrix, user_features_matrix, item_features_matrix,
                          extra_feed_kwargs=None):

        # Check that input data is of a sparse type
        if (interactions_matrix is not None) and (not sp.issparse(interactions_matrix)):
            raise Exception('Interactions must be a scipy sparse matrix')
        if not sp.issparse(user_features_matrix):
            raise Exception('User features must be a scipy sparse matrix')
        if not sp.issparse(item_features_matrix):
            raise Exception('Item features must be a scipy sparse matrix')

        # Create placeholders if interactions_matrix is none
        # TODO JK - This probably isn't necessary -- will the graph execute without it?
        if interactions_matrix is None:
            interactions_matrix = np.ones((user_features_matrix.shape[0], item_features_matrix.shape[0]))

        n_users, user_feature_indices, user_feature_values = self._process_matrix(user_features_matrix)
        n_items, item_feature_indices, item_feature_values = self._process_matrix(item_features_matrix)
        _, interaction_indices, interaction_values = self._process_matrix(interactions_matrix)

        feed_dict = {self.tf_n_users: n_users,
                     self.tf_n_items: n_items,
                     self.tf_user_feature_indices: user_feature_indices,
                     self.tf_user_feature_values: user_feature_values,
                     self.tf_item_feature_indices: item_feature_indices,
                     self.tf_item_feature_values: item_feature_values,
                     self.tf_interaction_indices: interaction_indices,
                     self.tf_interaction_values: interaction_values}

        if extra_feed_kwargs:
            feed_dict.update(extra_feed_kwargs)

        return feed_dict

    def _create_user_feed_dict(self, user_features_matrix, extra_feed_kwargs=None):

        if not sp.issparse(user_features_matrix):
            raise Exception('User features must be a scipy sparse matrix')

        n_users, user_feature_indices, user_feature_values = self._process_matrix(user_features_matrix)

        feed_dict = {self.tf_n_users: n_users,
                     self.tf_user_feature_indices: user_feature_indices,
                     self.tf_user_feature_values: user_feature_values}

        if extra_feed_kwargs:
            feed_dict.update(extra_feed_kwargs)

        return feed_dict

    def _create_item_feed_dict(self, item_features_matrix, extra_feed_kwargs=None):

        if not sp.issparse(item_features_matrix):
            raise Exception('Item features must be a scipy sparse matrix')

        n_items, item_feature_indices, item_feature_values = self._process_matrix(item_features_matrix)

        feed_dict = {self.tf_n_items: n_items,
                     self.tf_item_feature_indices: item_feature_indices,
                     self.tf_item_feature_values: item_feature_values}

        if extra_feed_kwargs:
            feed_dict.update(extra_feed_kwargs)

        return feed_dict

    def _process_matrix(self, features_matrix):

        if not isinstance(features_matrix, sp.coo_matrix):
            features_matrix = sp.coo_matrix(features_matrix)

        # "Actors" is used to signify "users or items" -- unused for interactions
        n_actors = features_matrix.shape[0]
        feature_indices = [pair for pair in six.moves.zip(features_matrix.row, features_matrix.col)]
        feature_values = features_matrix.data

        return n_actors, feature_indices, feature_values

    def _build_tf_graph(self, n_user_features, n_item_features):

        # Initialize placeholder values for inputs
        self.tf_n_users = tf.placeholder('int64')
        self.tf_n_items = tf.placeholder('int64')
        self.tf_user_feature_indices = tf.placeholder('int64', [None, 2])
        self.tf_user_feature_values = tf.placeholder('float', [None])
        self.tf_item_feature_indices = tf.placeholder('int64', [None, 2])
        self.tf_item_feature_values = tf.placeholder('float', [None])
        self.tf_interaction_indices = tf.placeholder('int64', [None, 2])
        self.tf_interaction_values = tf.placeholder('float', [None])
        self.tf_learning_rate = tf.placeholder('float', None)
        self.tf_alpha = tf.placeholder('float', None)

        # Construct the features and interactions as sparse matrices
        self.tf_user_features = tf.SparseTensor(self.tf_user_feature_indices, self.tf_user_feature_values,
                                                [self.tf_n_users, n_user_features])
        self.tf_item_features = tf.SparseTensor(self.tf_item_feature_indices, self.tf_item_feature_values,
                                                [self.tf_n_items, n_item_features])
        self.tf_interactions = tf.SparseTensor(self.tf_interaction_indices, self.tf_interaction_values,
                                               [self.tf_n_users, self.tf_n_items])

        # Build the representations
        self.tf_user_representation, user_weights = \
            self.user_repr_graph_factory(tf_features=self.tf_user_features,
                                         n_components=self.n_components,
                                         n_features=n_user_features,
                                         node_name_ending='user')
        self.tf_item_representation, item_weights = \
            self.item_repr_graph_factory(tf_features=self.tf_item_features,
                                         n_components=self.n_components,
                                         n_features=n_item_features,
                                         node_name_ending='item')

        # Calculate the user and item biases
        if self.biased:
            self.tf_user_feature_biases = tf.Variable(tf.zeros([n_user_features, 1]))
            self.tf_item_feature_biases = tf.Variable(tf.zeros([n_item_features, 1]))

            # The reduce sum is to perform a rank reduction
            self.tf_projected_user_biases = tf.reduce_sum(
                tf.sparse_tensor_dense_matmul(self.tf_user_features, self.tf_user_feature_biases),
                axis=1
            )
            self.tf_projected_item_biases = tf.reduce_sum(
                tf.sparse_tensor_dense_matmul(self.tf_item_features, self.tf_item_feature_biases),
                axis=1
            )

        # Prediction = user_repr * item_repr + user_bias + item_bias
        # For the parallel prediction case, repr matrices can be multiplied together and the projected biases can be
        # broadcast across the resultant matrix
        self.tf_prediction = tf.matmul(self.tf_user_representation, self.tf_item_representation, transpose_b=True)

        # Add biases, if this is a biased estimator
        if self.biased:
            self.tf_prediction = self.tf_prediction \
                                 + tf.expand_dims(self.tf_projected_user_biases, 1) \
                                 + tf.expand_dims(self.tf_projected_item_biases, 0)

        # For the serial prediction case, gather the desired values from the parallel prediction and the interactions
        self.tf_prediction_serial = tf.gather_nd(self.tf_prediction, indices=self.tf_interactions.indices)
        self.tf_interactions_serial = self.tf_interactions.values

        # Double-sortation serves as a ranking process
        # The +1 is so the top-ranked has a non-zero rank
        tf_prediction_item_size = tf.shape(self.tf_prediction)[1]
        tf_indices_of_ranks = tf.nn.top_k(self.tf_prediction, k=tf_prediction_item_size)[1]
        self.tf_rankings = tf.nn.top_k(-tf_indices_of_ranks, k=tf_prediction_item_size)[1] + 1

        self.tf_weights = []
        self.tf_weights.extend(user_weights)
        self.tf_weights.extend(item_weights)

        if self.biased:
            self.tf_weights.append(self.tf_user_feature_biases)
            self.tf_weights.append(self.tf_item_feature_biases)

        # Loss function nodes
        self.tf_basic_loss = self.loss_graph_factory(tf_prediction_serial=self.tf_prediction_serial,
                                                     tf_interactions_serial=self.tf_interactions_serial,
                                                     tf_prediction=self.tf_prediction,
                                                     tf_interactions=self.tf_interactions,
                                                     tf_rankings=self.tf_rankings)
        self.tf_weight_reg_loss = sum(tf.nn.l2_loss(weights) for weights in self.tf_weights)
        self.tf_loss = self.tf_basic_loss + (self.tf_alpha * self.tf_weight_reg_loss)
        self.tf_optimizer = tf.train.AdamOptimizer(learning_rate=self.tf_learning_rate).minimize(self.tf_loss)

    def fit(self, interactions, user_features, item_features, epochs=100, learning_rate=0.1, alpha=0.0001,
            verbose=False, out_sample_interactions=None):
        """
        Constructs the TensorRec graph and fits the model.
        :param interactions: scipy.sparse matrix
        A matrix of interactions of shape [n_users, n_items].
        :param user_features: scipy.sparse matrix
        A matrix of user features of shape [n_users, n_user_features].
        :param item_features: scipy.sparse matrix
        A matrix of item features of shape [n_items, n_item_features].
        :param epochs: Integer
        The number of epochs to fit the model.
        :param learning_rate: Float
        The learning rate of the model.
        :param alpha:
        The weight regularization loss coefficient.
        :param verbose: boolean
        If true, the model will print a number of status statements during fitting.
        :param out_sample_interactions: scipy.sparse matrix
        A matrix of interactions of shape [n_users, n_items].
        If not None, and verbose == True, the model will be evaluated on these interactions on every epoch.
        """

        # Pass-through to fit_partial
        self.fit_partial(interactions, user_features, item_features, epochs, learning_rate, alpha, verbose,
                         out_sample_interactions)

    def fit_partial(self, interactions, user_features, item_features, epochs=1, learning_rate=0.1,
                    alpha=0.0001, verbose=False, out_sample_interactions=None):
        """
        Constructs the TensorRec graph and fits the model.
        :param interactions: scipy.sparse matrix
        A matrix of interactions of shape [n_users, n_items].
        :param user_features: scipy.sparse matrix
        A matrix of user features of shape [n_users, n_user_features].
        :param item_features: scipy.sparse matrix
        A matrix of item features of shape [n_items, n_item_features].
        :param epochs: Integer
        The number of epochs to fit the model.
        :param learning_rate: Float
        The learning rate of the model.
        :param alpha:
        The weight regularization loss coefficient.
        :param verbose: boolean
        If true, the model will print a number of status statements during fitting.
        :param out_sample_interactions: scipy.sparse matrix
        A matrix of interactions of shape [n_users, n_items].
        If not None, and verbose == True, the model will be evaluated on these interactions on every epoch.
        """

        session = get_session()

        # Check if the graph has been constructed by checking the dense prediction node
        # If it hasn't been constructed, initialize it
        if self.tf_prediction is None:
            # Numbers of features are learned at fit time from the shape of these two matrices and cannot be changed
            # without refitting
            self._build_tf_graph(n_user_features=user_features.shape[1], n_item_features=item_features.shape[1])
            session.run(tf.global_variables_initializer())

        if verbose:
            print('Processing interaction and feature data')

        feed_dict = self._create_feed_dict(interactions, user_features, item_features,
                                           extra_feed_kwargs={self.tf_learning_rate: learning_rate,
                                                              self.tf_alpha: alpha})

        if verbose:
            print('Beginning fitting')

        for epoch in range(epochs):

            session.run(self.tf_optimizer, feed_dict=feed_dict)

            if verbose:
                mean_loss = self.tf_basic_loss.eval(session=session, feed_dict=feed_dict)
                mean_pred = np.mean(self.tf_prediction_serial.eval(session=session, feed_dict=feed_dict))
                weight_reg_l2_loss = (alpha * self.tf_weight_reg_loss).eval(session=session, feed_dict=feed_dict)
                print('EPOCH %s loss = %s, weight_reg_l2_loss = %s, mean_pred = %s' % (epoch, mean_loss,
                                                                                       weight_reg_l2_loss, mean_pred))
                if out_sample_interactions:
                    os_feed_dict = self._create_feed_dict(out_sample_interactions, user_features, item_features)
                    os_loss = self.tf_basic_loss.eval(session=session, feed_dict=os_feed_dict)
                    print('Out-Sample loss = %s' % os_loss)

    def predict(self, user_features, item_features):
        """
        Predict recommendation scores for the given users and items.
        :param user_features: scipy.sparse matrix
        A matrix of user features of shape [n_users, n_user_features].
        :param item_features: scipy.sparse matrix
        A matrix of item features of shape [n_items, n_item_features].
        :return: TBD
        """
        feed_dict = self._create_feed_dict(interactions_matrix=None,
                                           user_features_matrix=user_features,
                                           item_features_matrix=item_features)

        predictions = self.tf_prediction.eval(session=get_session(), feed_dict=feed_dict)

        return predictions

    def predict_rank(self, user_features, item_features):
        feed_dict = self._create_feed_dict(interactions_matrix=None,
                                           user_features_matrix=user_features,
                                           item_features_matrix=item_features)

        rankings = self.tf_rankings.eval(session=get_session(), feed_dict=feed_dict)

        return rankings

    def predict_user_representation(self, user_features):

        if self.biased:
            raise NotImplementedError('predict_user_representation() is not supported with biased models.'
                                      'Try TensorRec(biased=False)')

        feed_dict = self._create_user_feed_dict(user_features_matrix=user_features)
        user_repr = self.tf_user_representation.eval(session=get_session(), feed_dict=feed_dict)
        return user_repr

    def predict_item_representation(self, item_features):

        if self.biased:
            raise NotImplementedError('predict_item_representation() is not supported with biased models.'
                                      'Try TensorRec(biased=False)')

        feed_dict = self._create_item_feed_dict(item_features_matrix=item_features)
        item_repr = self.tf_item_representation.eval(session=get_session(), feed_dict=feed_dict)
        return item_repr
