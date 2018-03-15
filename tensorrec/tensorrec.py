import logging
import numpy as np
import pickle
from scipy import sparse as sp
import six
import tensorflow as tf

from .loss_graphs import AbstractLossGraph, RMSELossGraph
from .prediction_graphs import (
    AbstractPredictionGraph, DotProductPredictionGraph, CosineSimilarityPredictionGraph,
    EuclidianSimilarityPredictionGraph
)
from .recommendation_graphs import (
    project_biases, split_sparse_tensor_indices, bias_prediction_dense, bias_prediction_serial, rank_predictions,
    densify_sampled_item_predictions, collapse_mixture_of_tastes
)
from .representation_graphs import AbstractRepresentationGraph, LinearRepresentationGraph
from .session_management import get_session
from .util import sample_items, calculate_batched_alpha


class TensorRec(object):

    def __init__(self,
                 n_components=100,
                 n_tastes=1,
                 user_repr_graph=LinearRepresentationGraph(),
                 item_repr_graph=LinearRepresentationGraph(),
                 prediction_graph=DotProductPredictionGraph(),
                 loss_graph=RMSELossGraph(),
                 biased=True,
                 normalize_users=False,
                 normalize_items=False):
        """
        A TensorRec recommendation model.
        :param n_components: Integer
        The dimension of a single output of the representation function. Must be >= 1.
        :param n_tastes: Integer
        The number of tastes/reprs to be calculated for each user. Must be >= 1.
        :param user_repr_graph: AbstractRepresentationGraph
        An object which inherits AbstractRepresentationGraph that contains a method to calculate user representations.
        See tensorrec.representation_graphs for examples.
        :param item_repr_graph: AbstractRepresentationGraph
        An object which inherits AbstractRepresentationGraph that contains a method to calculate item representations.
        See tensorrec.representation_graphs for examples.
        :param prediction_graph: AbstractPredictionGraph
        An object which inherits AbstractPredictionGraph that contains a method to calculate predictions from a pair of
        user/item reprs.
        See tensorrec.prediction_graphs for examples.
        :param loss_graph: AbstractLossGraph
        An object which inherits AbstractLossGraph that contains a method to calculate the loss function.
        See tensorrec.loss_graphs for examples.
        :param biased: bool
        If True, a bias value will be calculated for every user feature and item feature.
        :param normalize_users: bool
        If True, every row of the user features will be l2 normalized.
        :param normalize_items: bool
        If True, every row of the item features will be l2 normalized.
        """

        # Arg-check
        if (n_components is None) or (n_tastes is None) or (user_repr_graph is None) or (item_repr_graph is None) \
                or (prediction_graph is None) or (loss_graph is None):
            raise ValueError("All arguments to TensorRec() must be non-None")
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        if n_tastes < 1:
            raise ValueError("n_tastes must be >= 1")
        if not isinstance(user_repr_graph, AbstractRepresentationGraph):
            raise ValueError("user_repr_graph must inherit AbstractRepresentationGraph")
        if not isinstance(item_repr_graph, AbstractRepresentationGraph):
            raise ValueError("item_repr_graph must inherit AbstractRepresentationGraph")
        if not isinstance(prediction_graph, AbstractPredictionGraph):
            raise ValueError("prediction_graph must inherit AbstractPredictionGraph")
        if not isinstance(loss_graph, AbstractLossGraph):
            raise ValueError("loss_graph must inherit AbstractLossGraph")

        self.n_components = n_components
        self.n_tastes = n_tastes
        self.user_repr_graph_factory = user_repr_graph
        self.item_repr_graph_factory = item_repr_graph
        self.prediction_graph_factory = prediction_graph
        self.loss_graph_factory = loss_graph
        self.biased = biased
        self.normalize_users = normalize_users
        self.normalize_items = normalize_items

        # A list of the attr names of every graph hook attr
        self.graph_tensor_hook_attr_names = [

            # Top-level API nodes
            'tf_user_representation', 'tf_item_representation', 'tf_prediction_serial', 'tf_prediction', 'tf_rankings',
            'tf_predict_dot_product', 'tf_predict_cosine_similarity', 'tf_predict_euclidian_similarity',

            # Training nodes
            'tf_basic_loss', 'tf_weight_reg_loss', 'tf_loss',

            # Feed placeholders
            'tf_n_users', 'tf_n_items', 'tf_user_feature_indices', 'tf_user_feature_values', 'tf_item_feature_indices',
            'tf_item_feature_values', 'tf_interaction_indices', 'tf_interaction_values', 'tf_learning_rate', 'tf_alpha',
            'tf_sample_indices', 'tf_n_sampled_items'
        ]
        if self.biased:
            self.graph_tensor_hook_attr_names += [
                'tf_projected_user_biases', 'tf_projected_item_biases',
            ]
        self.graph_operation_hook_attr_names = [

            # AdamOptimizer
            'tf_optimizer',

        ]
        self._clear_graph_hook_attrs()

        # A map of every graph hook attr name to the node name after construction
        # Tensors and operations are stored separated because they are handled differently by TensorFlow
        self.graph_tensor_hook_node_names = {}
        self.graph_operation_hook_node_names = {}

    def _clear_graph_hook_attrs(self):
        for graph_tensor_hook_attr_name in self.graph_tensor_hook_attr_names:
            self.__setattr__(graph_tensor_hook_attr_name, None)
        for graph_operation_hook_attr_name in self.graph_operation_hook_attr_names:
            self.__setattr__(graph_operation_hook_attr_name, None)

    def _attach_graph_hook_attrs(self):
        session = get_session()

        for graph_tensor_hook_attr_name in self.graph_tensor_hook_attr_names:
            graph_tensor_hook_node_name = self.graph_tensor_hook_node_names[graph_tensor_hook_attr_name]
            node = session.graph.get_tensor_by_name(name=graph_tensor_hook_node_name)
            self.__setattr__(graph_tensor_hook_attr_name, node)

        for graph_operation_hook_attr_name in self.graph_operation_hook_attr_names:
            graph_operation_hook_node_name = self.graph_operation_hook_node_names[graph_operation_hook_attr_name]
            node = session.graph.get_operation_by_name(name=graph_operation_hook_node_name)
            self.__setattr__(graph_operation_hook_attr_name, node)

    def _create_feed_dict(self, interactions_matrix, user_features_matrix, item_features_matrix,
                          extra_feed_kwargs=None):

        # Check that input data is of a sparse type
        if (interactions_matrix is not None) and (not sp.issparse(interactions_matrix)):
            raise Exception('Interactions must be a scipy sparse matrix')
        if not sp.issparse(user_features_matrix):
            raise Exception('User features must be a scipy sparse matrix')
        if not sp.issparse(item_features_matrix):
            raise Exception('Item features must be a scipy sparse matrix')

        n_users, user_feature_indices, user_feature_values = self._process_matrix(user_features_matrix,
                                                                                  normalize_rows=self.normalize_users)
        n_items, item_feature_indices, item_feature_values = self._process_matrix(item_features_matrix,
                                                                                  normalize_rows=self.normalize_items)

        feed_dict = {self.tf_n_users: n_users,
                     self.tf_n_items: n_items,
                     self.tf_user_feature_indices: user_feature_indices,
                     self.tf_user_feature_values: user_feature_values,
                     self.tf_item_feature_indices: item_feature_indices,
                     self.tf_item_feature_values: item_feature_values, }

        if interactions_matrix is not None:
            _, interaction_indices, interaction_values = self._process_matrix(interactions_matrix)
            feed_dict[self.tf_interaction_indices] = interaction_indices
            feed_dict[self.tf_interaction_values] = interaction_values

        if extra_feed_kwargs:
            feed_dict.update(extra_feed_kwargs)

        return feed_dict

    def _create_user_feed_dict(self, user_features_matrix, extra_feed_kwargs=None):

        if not sp.issparse(user_features_matrix):
            raise Exception('User features must be a scipy sparse matrix')

        n_users, user_feature_indices, user_feature_values = self._process_matrix(user_features_matrix,
                                                                                  normalize_rows=self.normalize_users)

        feed_dict = {self.tf_n_users: n_users,
                     self.tf_user_feature_indices: user_feature_indices,
                     self.tf_user_feature_values: user_feature_values}

        if extra_feed_kwargs:
            feed_dict.update(extra_feed_kwargs)

        return feed_dict

    def _create_item_feed_dict(self, item_features_matrix, extra_feed_kwargs=None):

        if not sp.issparse(item_features_matrix):
            raise Exception('Item features must be a scipy sparse matrix')

        n_items, item_feature_indices, item_feature_values = self._process_matrix(item_features_matrix,
                                                                                  normalize_rows=self.normalize_items)

        feed_dict = {self.tf_n_items: n_items,
                     self.tf_item_feature_indices: item_feature_indices,
                     self.tf_item_feature_values: item_feature_values}

        if extra_feed_kwargs:
            feed_dict.update(extra_feed_kwargs)

        return feed_dict

    def _process_matrix(self, features_matrix, normalize_rows=False):

        if normalize_rows:
            if not isinstance(features_matrix, sp.csr_matrix):
                features_matrix = sp.csr_matrix(features_matrix)
            mag = np.sqrt(features_matrix.power(2).sum(axis=1))
            features_matrix = features_matrix.multiply(1.0 / mag)

        if not isinstance(features_matrix, sp.coo_matrix):
            features_matrix = sp.coo_matrix(features_matrix)

        # "Actors" is used to signify "users or items" -- unused for interactions
        n_actors = features_matrix.shape[0]
        feature_indices = [pair for pair in six.moves.zip(features_matrix.row, features_matrix.col)]
        feature_values = features_matrix.data

        return n_actors, feature_indices, feature_values

    def _create_batch_feed_dicts(self, interactions, user_features, item_features, extra_feed_kwargs,
                                 user_batch_size=None):

        if not isinstance(interactions, sp.csr_matrix):
            interactions = sp.csr_matrix(interactions)
        if not isinstance(user_features, sp.csr_matrix):
            user_features = sp.csr_matrix(user_features)

        n_users = user_features.shape[0]

        # Infer the batch size, if necessary
        if user_batch_size is None:
            user_batch_size = n_users

        feed_dicts = []

        start_batch = 0
        while start_batch < n_users:

            # min() ensures that the batch bounds doesn't go past the end of the index
            end_batch = min(start_batch + user_batch_size, n_users)

            batch_interactions = interactions[start_batch:end_batch]
            batch_user_features = user_features[start_batch:end_batch]

            # TODO its inefficient to make so many copies of the item features
            feed_dict = self._create_feed_dict(interactions_matrix=batch_interactions,
                                               user_features_matrix=batch_user_features,
                                               item_features_matrix=item_features,
                                               extra_feed_kwargs=extra_feed_kwargs)
            feed_dicts.append(feed_dict)

            start_batch = end_batch

        return feed_dicts

    def _build_tf_graph(self, n_user_features, n_item_features):

        # Initialize placeholder values for inputs
        self.tf_n_users = tf.placeholder('int64')
        self.tf_n_items = tf.placeholder('int64')
        self.tf_n_sampled_items = tf.placeholder('int64')

        # SparseTensor placeholders
        self.tf_user_feature_indices = tf.placeholder('int64', [None, 2])
        self.tf_user_feature_values = tf.placeholder('float', [None])
        self.tf_item_feature_indices = tf.placeholder('int64', [None, 2])
        self.tf_item_feature_values = tf.placeholder('float', [None])
        self.tf_interaction_indices = tf.placeholder('int64', [None, 2])
        self.tf_interaction_values = tf.placeholder('float', [None])

        self.tf_sample_indices = tf.placeholder('int64', [None, None])
        self.tf_learning_rate = tf.placeholder('float', None)
        self.tf_alpha = tf.placeholder('float', None)

        # Construct the features and interactions as sparse matrices
        tf_user_features = tf.SparseTensor(self.tf_user_feature_indices, self.tf_user_feature_values,
                                           [self.tf_n_users, n_user_features])
        tf_item_features = tf.SparseTensor(self.tf_item_feature_indices, self.tf_item_feature_values,
                                           [self.tf_n_items, n_item_features])
        tf_interactions = tf.SparseTensor(self.tf_interaction_indices, self.tf_interaction_values,
                                          [self.tf_n_users, self.tf_n_items])

        # Collect the weights for normalization
        tf_weights = []

        # Build the item representations
        self.tf_item_representation, item_weights = \
            self.item_repr_graph_factory.connect_representation_graph(tf_features=tf_item_features,
                                                                      n_components=self.n_components,
                                                                      n_features=n_item_features,
                                                                      node_name_ending='item')
        tf_weights.extend(item_weights)

        tf_x_user, tf_x_item = split_sparse_tensor_indices(tf_sparse_tensor=tf_interactions, n_dimensions=2)
        tf_transposed_sample_indices = tf.transpose(self.tf_sample_indices)
        tf_x_user_sample = tf_transposed_sample_indices[0]
        tf_x_item_sample = tf_transposed_sample_indices[1]

        # Build n_tastes user representations and predictions
        tastes_tf_user_representations = []
        tastes_tf_predictions = []
        tastes_tf_prediction_serials = []
        tastes_tf_sample_prediction_serials = []
        tastes_tf_dot_products = []
        tastes_tf_cosine_sims = []
        tastes_tf_euclidian_sims = []

        for taste in range(self.n_tastes):
            tf_user_representation, user_weights = \
                self.user_repr_graph_factory.connect_representation_graph(tf_features=tf_user_features,
                                                                          n_components=self.n_components,
                                                                          n_features=n_user_features,
                                                                          node_name_ending='user_{}'.format(taste))
            tastes_tf_user_representations.append(tf_user_representation)
            tf_weights.extend(user_weights)

            # Connect the configurable prediction graphs for each taste
            tf_prediction = self.prediction_graph_factory.connect_dense_prediction_graph(
                tf_user_representation=tf_user_representation,
                tf_item_representation=self.tf_item_representation
            )
            tf_prediction_serial = self.prediction_graph_factory.connect_serial_prediction_graph(
                tf_user_representation=tf_user_representation,
                tf_item_representation=self.tf_item_representation,
                tf_x_user=tf_x_user,
                tf_x_item=tf_x_item,
            )
            tf_sample_predictions_serial = self.prediction_graph_factory.connect_serial_prediction_graph(
                tf_user_representation=tf_user_representation,
                tf_item_representation=self.tf_item_representation,
                tf_x_user=tf_x_user_sample,
                tf_x_item=tf_x_item_sample,
            )
            tf_predict_dot_product = DotProductPredictionGraph().connect_dense_prediction_graph(
                tf_user_representation=tf_user_representation,
                tf_item_representation=self.tf_item_representation,
            )
            tf_predict_cosine_similarity = CosineSimilarityPredictionGraph().connect_dense_prediction_graph(
                tf_user_representation=tf_user_representation,
                tf_item_representation=self.tf_item_representation,
            )
            tf_predict_euclidian_similarity = EuclidianSimilarityPredictionGraph().connect_dense_prediction_graph(
                tf_user_representation=tf_user_representation,
                tf_item_representation=self.tf_item_representation,
            )

            # Append to tastes
            tastes_tf_predictions.append(tf_prediction)
            tastes_tf_prediction_serials.append(tf_prediction_serial)
            tastes_tf_sample_prediction_serials.append(tf_sample_predictions_serial)
            tastes_tf_dot_products.append(tf_predict_dot_product)
            tastes_tf_cosine_sims.append(tf_predict_cosine_similarity)
            tastes_tf_euclidian_sims.append(tf_predict_euclidian_similarity)

        self.tf_user_representation = tf.stack(tastes_tf_user_representations)
        self.tf_prediction = collapse_mixture_of_tastes(tastes_tf_predictions)
        self.tf_prediction_serial = collapse_mixture_of_tastes(tastes_tf_prediction_serials)
        tf_sample_predictions_serial = collapse_mixture_of_tastes(tastes_tf_sample_prediction_serials)

        # Add biases, if this is a biased estimator
        if self.biased:
            tf_user_feature_biases, self.tf_projected_user_biases = project_biases(
                tf_features=tf_user_features, n_features=n_user_features
            )
            tf_item_feature_biases, self.tf_projected_item_biases = project_biases(
                tf_features=tf_item_features, n_features=n_item_features
            )

            tf_weights.append(tf_user_feature_biases)
            tf_weights.append(tf_item_feature_biases)

            self.tf_prediction = bias_prediction_dense(
                tf_prediction=self.tf_prediction,
                tf_projected_user_biases=self.tf_projected_user_biases,
                tf_projected_item_biases=self.tf_projected_item_biases)

            self.tf_prediction_serial = bias_prediction_serial(
                tf_prediction_serial=self.tf_prediction_serial,
                tf_projected_user_biases=self.tf_projected_user_biases,
                tf_projected_item_biases=self.tf_projected_item_biases,
                tf_x_user=tf_x_user,
                tf_x_item=tf_x_item)

            tf_sample_predictions_serial = bias_prediction_serial(
                tf_prediction_serial=tf_sample_predictions_serial,
                tf_projected_user_biases=self.tf_projected_user_biases,
                tf_projected_item_biases=self.tf_projected_item_biases,
                tf_x_user=tf_x_user_sample,
                tf_x_item=tf_x_item_sample)

        tf_interactions_serial = tf_interactions.values

        # Construct API nodes
        self.tf_rankings = rank_predictions(tf_prediction=self.tf_prediction)
        self.tf_predict_dot_product = collapse_mixture_of_tastes(tastes_tf_dot_products)
        self.tf_predict_cosine_similarity = collapse_mixture_of_tastes(tastes_tf_cosine_sims)
        self.tf_predict_euclidian_similarity = collapse_mixture_of_tastes(tastes_tf_euclidian_sims)

        # Compose loss function args
        # This composition is for execution safety: it prevents loss functions that are incorrectly configured from
        # having visibility of certain nodes.
        loss_graph_kwargs = {
            'tf_prediction_serial': self.tf_prediction_serial,
            'tf_interactions_serial': tf_interactions_serial,
            'tf_interactions': tf_interactions,
            'tf_n_users': self.tf_n_users,
            'tf_n_items': self.tf_n_items,
        }
        if self.loss_graph_factory.is_dense:
            loss_graph_kwargs.update({
                'tf_prediction': self.tf_prediction,
                'tf_rankings': self.tf_rankings,
            })
        if self.loss_graph_factory.is_sample_based:
            tf_sample_predictions = densify_sampled_item_predictions(
                tf_sample_predictions_serial=tf_sample_predictions_serial,
                tf_n_sampled_items=self.tf_n_sampled_items,
                tf_n_users=self.tf_n_users,
            )
            loss_graph_kwargs.update({'tf_sample_predictions': tf_sample_predictions,
                                      'tf_n_sampled_items': self.tf_n_sampled_items})

        # Build loss graph
        self.tf_basic_loss = self.loss_graph_factory.connect_loss_graph(**loss_graph_kwargs)

        self.tf_weight_reg_loss = sum(tf.nn.l2_loss(weights) for weights in tf_weights)
        self.tf_loss = self.tf_basic_loss + (self.tf_alpha * self.tf_weight_reg_loss)
        self.tf_optimizer = tf.train.AdamOptimizer(learning_rate=self.tf_learning_rate).minimize(self.tf_loss)

        # Get node names for each graph hook
        for graph_tensor_hook_attr_name in self.graph_tensor_hook_attr_names:
            hook = self.__getattribute__(graph_tensor_hook_attr_name)
            self.graph_tensor_hook_node_names[graph_tensor_hook_attr_name] = hook.name
        for graph_operation_hook_attr_name in self.graph_operation_hook_attr_names:
            hook = self.__getattribute__(graph_operation_hook_attr_name)
            self.graph_operation_hook_node_names[graph_operation_hook_attr_name] = hook.name

    def fit(self, interactions, user_features, item_features, epochs=100, learning_rate=0.1, alpha=0.00001,
            verbose=False, out_sample_interactions=None, user_batch_size=None, n_sampled_items=None):
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
        :param user_batch_size: int or None
        The maximum number of users per batch, or None for all users.
        :param n_sampled_items: int or None
        The number of items to sample per user for use in loss functions. Must be non-None if
        self.loss_graph_factory.is_sample_based is True.
        """

        # Pass-through to fit_partial
        self.fit_partial(interactions=interactions,
                         user_features=user_features,
                         item_features=item_features,
                         epochs=epochs,
                         learning_rate=learning_rate,
                         alpha=alpha,
                         verbose=verbose,
                         out_sample_interactions=out_sample_interactions,
                         user_batch_size=user_batch_size,
                         n_sampled_items=n_sampled_items)

    def fit_partial(self, interactions, user_features, item_features, epochs=1, learning_rate=0.1,
                    alpha=0.00001, verbose=False, out_sample_interactions=None, user_batch_size=None,
                    n_sampled_items=None):
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
        :param user_batch_size: int or None
        The maximum number of users per batch, or None for all users.
        :param n_sampled_items: int or None
        The number of items to sample per user for use in loss functions. Must be non-None if
        self.loss_graph_factory.is_sample_based is True.
        """

        session = get_session()

        # Arg checking
        if self.loss_graph_factory.is_sample_based:
            if (n_sampled_items is None) or (n_sampled_items <= 0):
                raise ValueError("n_sampled_items must be an integer >0")
        if (n_sampled_items is not None) and (not self.loss_graph_factory.is_sample_based):
            logging.warning('n_sampled_items was specified, but the loss graph is not sample-based')

        # Check if the graph has been constructed by checking the dense prediction node
        # If it hasn't been constructed, initialize it
        if self.tf_prediction is None:
            # Numbers of features are learned at fit time from the shape of these two matrices and cannot be changed
            # without refitting
            self._build_tf_graph(n_user_features=user_features.shape[1], n_item_features=item_features.shape[1])
            session.run(tf.global_variables_initializer())

        if verbose:
            logging.info('Processing interaction and feature data')

        batched_feed_dicts = self._create_batch_feed_dicts(interactions=interactions,
                                                           user_features=user_features,
                                                           item_features=item_features,
                                                           user_batch_size=user_batch_size,
                                                           extra_feed_kwargs={self.tf_learning_rate: learning_rate})

        # This scales down the alpha based on the number of batches and inserts the new alpha in all feed dicts
        batched_alpha = calculate_batched_alpha(num_batches=len(batched_feed_dicts), alpha=alpha)
        for feed_dict in batched_feed_dicts:
            feed_dict[self.tf_alpha] = batched_alpha

        if verbose:
            logging.info('Beginning fitting')

        for epoch in range(epochs):
            for batch, feed_dict in enumerate(batched_feed_dicts):

                # Handle random item sampling, if applicable
                if self.loss_graph_factory.is_sample_based:
                    sample_indices = sample_items(n_users=feed_dict[self.tf_n_users],
                                                  n_items=feed_dict[self.tf_n_items],
                                                  n_sampled_items=n_sampled_items,
                                                  replace=self.loss_graph_factory.is_sampled_with_replacement)
                    feed_dict[self.tf_sample_indices] = sample_indices
                    feed_dict[self.tf_n_sampled_items] = n_sampled_items

                # TODO find something more elegant than these cascaded ifs
                if not verbose:
                    session.run(self.tf_optimizer, feed_dict=feed_dict)

                else:
                    _, loss, serial_predictions, wr_loss = session.run(
                        [self.tf_optimizer, self.tf_basic_loss, self.tf_prediction_serial, self.tf_weight_reg_loss],
                        feed_dict=feed_dict
                    )
                    mean_loss = np.mean(loss)
                    mean_pred = np.mean(serial_predictions)
                    weight_reg_l2_loss = alpha * wr_loss
                    logging.info('EPOCH {} BATCH {} loss = {}, weight_reg_l2_loss = {}, mean_pred = {}'.format(
                        epoch, batch, mean_loss, weight_reg_l2_loss, mean_pred
                    ))
                    if out_sample_interactions:
                        os_feed_dict = self._create_feed_dict(out_sample_interactions, user_features, item_features)
                        os_loss = self.tf_basic_loss.eval(session=session, feed_dict=os_feed_dict)
                        logging.info('Out-Sample loss = {}'.format(os_loss))

    def predict(self, user_features, item_features):
        """
        Predict recommendation scores for the given users and items.
        :param user_features: scipy.sparse matrix
        A matrix of user features of shape [n_users, n_user_features].
        :param item_features: scipy.sparse matrix
        A matrix of item features of shape [n_items, n_item_features].
        :return: np.ndarray
        The predictions in an ndarray of shape [n_users, n_items]
        """
        feed_dict = self._create_feed_dict(interactions_matrix=None,
                                           user_features_matrix=user_features,
                                           item_features_matrix=item_features)

        predictions = self.tf_prediction.eval(session=get_session(), feed_dict=feed_dict)

        return predictions

    def predict_dot_product(self, user_features, item_features):
        """
        Predicts the latent dot product between the given users and items.
        :param user_features: scipy.sparse matrix
        A matrix of user features of shape [n_users, n_user_features].
        :param item_features: scipy.sparse matrix
        A matrix of item features of shape [n_items, n_item_features].
        :return: np.ndarray
        The dot products in an ndarray of shape [n_users, n_items]
        """
        feed_dict = self._create_feed_dict(interactions_matrix=None,
                                           user_features_matrix=user_features,
                                           item_features_matrix=item_features)
        predictions = self.tf_predict_dot_product.eval(session=get_session(), feed_dict=feed_dict)
        return predictions

    def predict_cosine_similarity(self, user_features, item_features):
        """
        Predicts the latent cosine similarity between the given users and items.
        :param user_features: scipy.sparse matrix
        A matrix of user features of shape [n_users, n_user_features].
        :param item_features: scipy.sparse matrix
        A matrix of item features of shape [n_items, n_item_features].
        :return: np.ndarray
        The predictions in an ndarray of shape [n_users, n_items]
        """
        feed_dict = self._create_feed_dict(interactions_matrix=None,
                                           user_features_matrix=user_features,
                                           item_features_matrix=item_features)
        predictions = self.tf_predict_cosine_similarity.eval(session=get_session(), feed_dict=feed_dict)
        return predictions

    def predict_euclidian_similarity(self, user_features, item_features):
        """
        Predicts the latent euclidian similarity between the given users and items.
        :param user_features: scipy.sparse matrix
        A matrix of user features of shape [n_users, n_user_features].
        :param item_features: scipy.sparse matrix
        A matrix of item features of shape [n_items, n_item_features].
        :return: np.ndarray
        The predictions in an ndarray of shape [n_users, n_items]
        """
        feed_dict = self._create_feed_dict(interactions_matrix=None,
                                           user_features_matrix=user_features,
                                           item_features_matrix=item_features)
        predictions = self.tf_predict_euclidian_similarity.eval(session=get_session(), feed_dict=feed_dict)
        return predictions

    def predict_rank(self, user_features, item_features):
        """
        Predict recommendation ranks for the given users and items.
        :param user_features: scipy.sparse matrix
        A matrix of user features of shape [n_users, n_user_features].
        :param item_features: scipy.sparse matrix
        A matrix of item features of shape [n_items, n_item_features].
        :return: np.ndarray
        The ranks in an ndarray of shape [n_users, n_items]
        """
        feed_dict = self._create_feed_dict(interactions_matrix=None,
                                           user_features_matrix=user_features,
                                           item_features_matrix=item_features)

        rankings = self.tf_rankings.eval(session=get_session(), feed_dict=feed_dict)

        return rankings

    def predict_user_representation(self, user_features):
        """
        Predict latent representation vectors for the given users.
        :param user_features: scipy.sparse matrix
        A matrix of user features of shape [n_users, n_user_features].
        :return: np.ndarray
        The latent user representations in an ndarray of shape [n_users, n_components]
        """
        feed_dict = self._create_user_feed_dict(user_features_matrix=user_features)
        user_repr = self.tf_user_representation.eval(session=get_session(), feed_dict=feed_dict)

        # If there is only one user repr per user, collapse from rank 3 to rank 2
        if self.n_tastes == 1:
            user_repr = np.sum(user_repr, axis=0)

        return user_repr

    def predict_item_representation(self, item_features):
        """
        Predict representation vectors for the given items.
        :param item_features: scipy.sparse matrix
        A matrix of item features of shape [n_items, n_item_features].
        :return: np.ndarray
        The latent item representations in an ndarray of shape [n_items, n_components]
        """
        feed_dict = self._create_item_feed_dict(item_features_matrix=item_features)
        item_repr = self.tf_item_representation.eval(session=get_session(), feed_dict=feed_dict)
        return item_repr

    def predict_user_bias(self, user_features):
        """
        Predict bias values for the given users.
        :param user_features: scipy.sparse matrix
        A matrix of user features of shape [n_users, n_user_features].
        :return: np.ndarray
        The user biases in an ndarray of shape [n_users]
        """
        if not self.biased:
            raise NotImplementedError('Cannot predict user bias for unbiased model')
        feed_dict = self._create_user_feed_dict(user_features_matrix=user_features)
        predictions = self.tf_projected_user_biases.eval(session=get_session(), feed_dict=feed_dict)
        return predictions

    def predict_item_bias(self, item_features):
        """
        Predict bias values for the given items.
        :param item_features: scipy.sparse matrix
        A matrix of item features of shape [n_items, n_item_features].
        :return: np.ndarray
        The item biases in an ndarray of shape [n_items]
        """
        if not self.biased:
            raise NotImplementedError('Cannot predict item bias for unbiased model')
        feed_dict = self._create_item_feed_dict(item_features_matrix=item_features)
        predictions = self.tf_projected_item_biases.eval(session=get_session(), feed_dict=feed_dict)
        return predictions

    def save_model(self, directory_path):
        """
        Saves the model to files in the given directory.
        :param directory_path: str
        The path to the directory in which to save the model.
        :return:
        """

        saver = tf.train.Saver()
        session_path = '{}/tensorrec_session.cpkt'.format(directory_path)
        saver.save(sess=get_session(), save_path=session_path)

        # Break connections to the graph before saving the python object
        self._clear_graph_hook_attrs()
        tensorrec_path = '{}/tensorrec.pkl'.format(directory_path)
        with open(tensorrec_path, 'wb') as file:
            pickle.dump(file=file, obj=self)

        # Reconnect to the graph after saving
        self._attach_graph_hook_attrs()

    @classmethod
    def load_model(cls, directory_path):
        """
        Loads the TensorRec model and TensorFlow session saved in the given directory.
        :param directory_path: str
        The path to the directory containing the saved model.
        :return:
        """

        saver = tf.train.Saver()
        session_path = '{}/tensorrec_session.cpkt'.format(directory_path)
        saver.restore(sess=get_session(), save_path=session_path)

        tensorrec_path = '{}/tensorrec.pkl'.format(directory_path)
        with open(tensorrec_path, 'rb') as file:
            model = pickle.load(file=file)
        model._attach_graph_hook_attrs()
        return model
