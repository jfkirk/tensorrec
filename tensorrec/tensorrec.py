from functools import partial
from itertools import cycle
import logging
import numpy as np
import os
import pickle
from scipy import sparse as sp
import tensorflow as tf

from .errors import (
    ModelNotBiasedException, ModelNotFitException, ModelWithoutAttentionException, BatchNonSparseInputException,
    TfVersionException
)
from .input_utils import create_tensorrec_iterator, get_dimensions_from_tensorrec_dataset
from .loss_graphs import AbstractLossGraph, RMSELossGraph
from .prediction_graphs import AbstractPredictionGraph, DotProductPredictionGraph
from .recommendation_graphs import (
    project_biases, split_sparse_tensor_indices, bias_prediction_dense, bias_prediction_serial, rank_predictions,
    densify_sampled_item_predictions, collapse_mixture_of_tastes, predict_similar_items
)
from .representation_graphs import AbstractRepresentationGraph, LinearRepresentationGraph
from .session_management import get_session
from .util import sample_items, calculate_batched_alpha, datasets_from_raw_input


class TensorRec(object):

    def __init__(self,
                 n_components=100,
                 n_tastes=1,
                 user_repr_graph=LinearRepresentationGraph(),
                 item_repr_graph=LinearRepresentationGraph(),
                 attention_graph=None,
                 prediction_graph=DotProductPredictionGraph(),
                 loss_graph=RMSELossGraph(),
                 biased=True,):
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
        :param attention_graph: AbstractRepresentationGraph or None
        Optional. An object which inherits AbstractRepresentationGraph that contains a method to calculate user
        attention. Any valid repr_graph is also a valid attention graph. If None, no attention process will be applied.
        :param prediction_graph: AbstractPredictionGraph
        An object which inherits AbstractPredictionGraph that contains a method to calculate predictions from a pair of
        user/item reprs.
        See tensorrec.prediction_graphs for examples.
        :param loss_graph: AbstractLossGraph
        An object which inherits AbstractLossGraph that contains a method to calculate the loss function.
        See tensorrec.loss_graphs for examples.
        :param biased: bool
        If True, a bias value will be calculated for every user feature and item feature.
        """

        # Check TensorFlow version
        major, minor, patch = tf.__version__.split(".")
        if int(major) < 1 or int(major) == 1 and int(minor) < 7:
            raise TfVersionException(tf_version=tf.__version__)

        # Arg Check
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
        if attention_graph is not None:
            if not isinstance(attention_graph, AbstractRepresentationGraph):
                raise ValueError("attention_graph must be None or inherit AbstractRepresentationGraph")
            if n_tastes == 1:
                raise ValueError("attention_graph must be None if n_tastes == 1")

        self.n_components = n_components
        self.n_tastes = n_tastes
        self.user_repr_graph_factory = user_repr_graph
        self.item_repr_graph_factory = item_repr_graph
        self.attention_graph_factory = attention_graph
        self.prediction_graph_factory = prediction_graph
        self.loss_graph_factory = loss_graph
        self.biased = biased

        # A list of the attr names of every graph hook attr
        self.graph_tensor_hook_attr_names = [

            # Top-level API nodes
            'tf_user_representation', 'tf_item_representation', 'tf_prediction_serial', 'tf_prediction', 'tf_rankings',
            'tf_predict_similar_items', 'tf_rank_similar_items',

            # Training nodes
            'tf_basic_loss', 'tf_weight_reg_loss', 'tf_loss',

            # Feed placeholders
            'tf_learning_rate', 'tf_alpha', 'tf_sample_indices', 'tf_n_sampled_items', 'tf_similar_items_ids',
        ]
        if self.biased:
            self.graph_tensor_hook_attr_names += ['tf_projected_user_biases', 'tf_projected_item_biases']
        if self.attention_graph_factory is not None:
            self.graph_tensor_hook_attr_names += ['tf_user_attention_representation']

        self.graph_operation_hook_attr_names = [
            # AdamOptimizer
            'tf_optimizer',
        ]
        self.graph_iterator_hook_attr_names = [
            # Input data iterators
            'tf_user_feature_iterator', 'tf_item_feature_iterator', 'tf_interaction_iterator',
        ]

        # Calling the break routine during __init__ creates all the attrs on the TensorRec object with an initial value
        # of None
        self._break_graph_hooks()

        # A map of every graph hook attr name to the node name after construction
        # Tensors and operations are stored separated because they are handled differently by TensorFlow
        self.graph_tensor_hook_node_names = {}
        self.graph_operation_hook_node_names = {}
        self.graph_iterator_hook_node_names = {}

    def _break_graph_hooks(self):
        for graph_tensor_hook_attr_name in self.graph_tensor_hook_attr_names:
            self.__setattr__(graph_tensor_hook_attr_name, None)
        for graph_operation_hook_attr_name in self.graph_operation_hook_attr_names:
            self.__setattr__(graph_operation_hook_attr_name, None)
        for graph_iterator_hook_attr_name in self.graph_iterator_hook_attr_names:
            self.__setattr__(graph_iterator_hook_attr_name, None)

    def _attach_graph_hooks(self):
        session = get_session()

        for graph_tensor_hook_attr_name in self.graph_tensor_hook_attr_names:
            graph_tensor_hook_node_name = self.graph_tensor_hook_node_names[graph_tensor_hook_attr_name]
            node = session.graph.get_tensor_by_name(name=graph_tensor_hook_node_name)
            self.__setattr__(graph_tensor_hook_attr_name, node)

        for graph_operation_hook_attr_name in self.graph_operation_hook_attr_names:
            graph_operation_hook_node_name = self.graph_operation_hook_node_names[graph_operation_hook_attr_name]
            node = session.graph.get_operation_by_name(name=graph_operation_hook_node_name)
            self.__setattr__(graph_operation_hook_attr_name, node)

        for graph_iterator_hook_attr_name in self.graph_iterator_hook_attr_names:
            iterator_resource_name, output_types, output_shapes, output_classes = \
                self.graph_iterator_hook_node_names[graph_iterator_hook_attr_name]
            iterator_resource = session.graph.get_tensor_by_name(name=iterator_resource_name)
            iterator = tf.data.Iterator(iterator_resource, None, output_types, output_shapes, output_classes)
            self.__setattr__(graph_iterator_hook_attr_name, iterator)

    def _record_graph_hook_names(self):

        # Record serializable node names/info for each graph hook
        for graph_tensor_hook_attr_name in self.graph_tensor_hook_attr_names:
            hook = self.__getattribute__(graph_tensor_hook_attr_name)
            self.graph_tensor_hook_node_names[graph_tensor_hook_attr_name] = hook.name

        for graph_operation_hook_attr_name in self.graph_operation_hook_attr_names:
            hook = self.__getattribute__(graph_operation_hook_attr_name)
            self.graph_operation_hook_node_names[graph_operation_hook_attr_name] = hook.name

        for graph_iterator_hook_attr_name in self.graph_iterator_hook_attr_names:
            hook = self.__getattribute__(graph_iterator_hook_attr_name)
            iterator_resource_name = hook._iterator_resource.name
            output_types = hook._output_types
            output_shapes = hook._output_shapes
            output_classes = hook._output_classes
            self.graph_iterator_hook_node_names[graph_iterator_hook_attr_name] = (
                iterator_resource_name, output_types, output_shapes, output_classes
            )

    def _create_batched_dataset_initializers(self, interactions, user_features, item_features, user_batch_size=None):

        if user_batch_size is not None:

            # Raise exception if interactions and user_features aren't sparse matrices
            if (not sp.issparse(interactions)) or (not sp.issparse(user_features)):
                raise BatchNonSparseInputException()

            # Coerce to CSR for fast batching
            if not isinstance(interactions, sp.csr_matrix):
                interactions = sp.csr_matrix(interactions)
            if not isinstance(user_features, sp.csr_matrix):
                user_features = sp.csr_matrix(user_features)

            n_users = user_features.shape[0]

            # Infer the batch size, if necessary
            if user_batch_size is None:
                user_batch_size = n_users

            interactions_batched = []
            user_features_batched = []

            start_batch = 0
            while start_batch < n_users:

                # min() ensures that the batch bounds doesn't go past the end of the index
                end_batch = min(start_batch + user_batch_size, n_users)

                interactions_batched.append(interactions[start_batch:end_batch])
                user_features_batched.append(user_features[start_batch:end_batch])

                start_batch = end_batch

            # Overwrite the input with the new, batched input
            interactions = interactions_batched
            user_features = user_features_batched

        # TODO this is hand-wavy and begging for a cleaner refactor
        (int_ds, uf_ds, if_ds), (int_init, uf_init, if_init) = self._create_datasets_and_initializers(
            interactions=interactions, user_features=user_features, item_features=item_features
        )

        # Ensure that lengths make sense
        if len(int_init) != len(uf_init):
            raise ValueError('Number of batches in user_features and interactions must be equal.')
        if (len(if_init) > 1) and (len(if_init) != len(uf_init)):
            raise ValueError('Number of batches in item_features must be 1 or equal to the number of batches in '
                             'user_features.')

        # Cycle item features when zipping because there should only be one
        datasets = [ds_set for ds_set in zip(int_ds, uf_ds, cycle(if_ds))]
        initializers = [init_set for init_set in zip(int_init, uf_init, cycle(if_init))]

        return datasets, initializers

    def _create_datasets_and_initializers(self, interactions=None, user_features=None, item_features=None):

        datasets = []
        initializers = []

        if interactions is not None:
            interactions_datasets = datasets_from_raw_input(raw_input=interactions)
            interactions_initializers = [self.tf_interaction_iterator.make_initializer(dataset)
                                         for dataset in interactions_datasets]
            datasets.append(interactions_datasets)
            initializers.append(interactions_initializers)

        if user_features is not None:
            user_features_datasets = datasets_from_raw_input(raw_input=user_features)
            user_features_initializers = [self.tf_user_feature_iterator.make_initializer(dataset)
                                          for dataset in user_features_datasets]
            datasets.append(user_features_datasets)
            initializers.append(user_features_initializers)

        if item_features is not None:
            item_features_datasets = datasets_from_raw_input(raw_input=item_features)
            item_features_initializers = [self.tf_item_feature_iterator.make_initializer(dataset)
                                          for dataset in item_features_datasets]
            datasets.append(item_features_datasets)
            initializers.append(item_features_initializers)

        return datasets, initializers

    def _build_input_iterators(self):
        self.tf_user_feature_iterator = create_tensorrec_iterator(name='tf_user_feature_iterator')
        self.tf_item_feature_iterator = create_tensorrec_iterator(name='tf_item_feature_iterator')
        self.tf_interaction_iterator = create_tensorrec_iterator(name='tf_interaction_iterator')

    def _build_tf_graph(self, n_user_features, n_item_features):

        # Build placeholders
        self.tf_n_sampled_items = tf.placeholder('int64')
        self.tf_similar_items_ids = tf.placeholder('int64', [None])
        self.tf_learning_rate = tf.placeholder('float', None)
        self.tf_alpha = tf.placeholder('float', None)

        tf_user_feature_rows, tf_user_feature_cols, tf_user_feature_values, tf_n_users, _ = \
            self.tf_user_feature_iterator.get_next()
        tf_item_feature_rows, tf_item_feature_cols, tf_item_feature_values, tf_n_items, _ = \
            self.tf_item_feature_iterator.get_next()
        tf_interaction_rows, tf_interaction_cols, tf_interaction_values, _, _ = \
            self.tf_interaction_iterator.get_next()

        tf_user_feature_indices = tf.stack([tf_user_feature_rows, tf_user_feature_cols], axis=1)
        tf_item_feature_indices = tf.stack([tf_item_feature_rows, tf_item_feature_cols], axis=1)
        tf_interaction_indices = tf.stack([tf_interaction_rows, tf_interaction_cols], axis=1)

        # Construct the features and interactions as sparse matrices
        tf_user_features = tf.SparseTensor(tf_user_feature_indices, tf_user_feature_values,
                                           [tf_n_users, n_user_features])
        tf_item_features = tf.SparseTensor(tf_item_feature_indices, tf_item_feature_values,
                                           [tf_n_items, n_item_features])
        tf_interactions = tf.SparseTensor(tf_interaction_indices, tf_interaction_values,
                                          [tf_n_users, tf_n_items])

        # Construct the sampling py_func
        sample_items_partial = partial(sample_items, replace=self.loss_graph_factory.is_sampled_with_replacement)
        self.tf_sample_indices = tf.py_func(func=sample_items_partial,
                                            inp=[tf_n_items, tf_n_users, self.tf_n_sampled_items],
                                            Tout=tf.int64)
        self.tf_sample_indices.set_shape([None, None])

        # Collect the weights for regularization
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

        # These lists will hold the reprs and predictions for each taste
        tastes_tf_user_representations = []
        tastes_tf_predictions = []
        tastes_tf_prediction_serials = []
        tastes_tf_sample_prediction_serials = []

        # If this model does not use attention, Nones are used as sentinels in place of the attentions
        if self.attention_graph_factory is not None:
            tastes_tf_attentions = []
            tastes_tf_attention_serials = []
            tastes_tf_sample_attention_serials = []
            tastes_tf_attention_representations = []
        else:
            tastes_tf_attentions = None
            tastes_tf_attention_serials = None
            tastes_tf_sample_attention_serials = None
            tastes_tf_attention_representations = None

        # Build n_tastes user representations and predictions
        for taste in range(self.n_tastes):
            tf_user_representation, user_weights = \
                self.user_repr_graph_factory.connect_representation_graph(tf_features=tf_user_features,
                                                                          n_components=self.n_components,
                                                                          n_features=n_user_features,
                                                                          node_name_ending='user_{}'.format(taste))
            tastes_tf_user_representations.append(tf_user_representation)
            tf_weights.extend(user_weights)

            # Connect attention, if applicable
            if self.attention_graph_factory is not None:
                tf_attention_representation, attention_weights = \
                    self.attention_graph_factory.connect_representation_graph(tf_features=tf_user_features,
                                                                              n_components=self.n_components,
                                                                              n_features=n_user_features,
                                                                              node_name_ending='attn_{}'.format(taste))
                tf_weights.extend(attention_weights)

                tf_attention = self.prediction_graph_factory.connect_dense_prediction_graph(
                    tf_user_representation=tf_attention_representation,
                    tf_item_representation=self.tf_item_representation
                )
                tf_attention_serial = self.prediction_graph_factory.connect_serial_prediction_graph(
                    tf_user_representation=tf_attention_representation,
                    tf_item_representation=self.tf_item_representation,
                    tf_x_user=tf_x_user,
                    tf_x_item=tf_x_item,
                )
                tf_sample_attention_serial = self.prediction_graph_factory.connect_serial_prediction_graph(
                    tf_user_representation=tf_user_representation,
                    tf_item_representation=self.tf_item_representation,
                    tf_x_user=tf_x_user_sample,
                    tf_x_item=tf_x_item_sample,
                )

                tastes_tf_attentions.append(tf_attention)
                tastes_tf_attention_serials.append(tf_attention_serial)
                tastes_tf_sample_attention_serials.append(tf_sample_attention_serial)
                tastes_tf_attention_representations.append(tf_attention_representation)

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

            # Append to tastes
            tastes_tf_predictions.append(tf_prediction)
            tastes_tf_prediction_serials.append(tf_prediction_serial)
            tastes_tf_sample_prediction_serials.append(tf_sample_predictions_serial)

        # If attention is in the graph, build the API node
        if self.attention_graph_factory is not None:
            self.tf_user_attention_representation = tf.stack(tastes_tf_attention_representations)

        self.tf_user_representation = tf.stack(tastes_tf_user_representations)
        self.tf_prediction = collapse_mixture_of_tastes(
            tastes_predictions=tastes_tf_predictions,
            tastes_attentions=tastes_tf_attentions
        )
        self.tf_prediction_serial = collapse_mixture_of_tastes(
            tastes_predictions=tastes_tf_prediction_serials,
            tastes_attentions=tastes_tf_attention_serials
        )
        tf_sample_predictions_serial = collapse_mixture_of_tastes(
            tastes_predictions=tastes_tf_sample_prediction_serials,
            tastes_attentions=tastes_tf_sample_attention_serials
        )

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
        self.tf_predict_similar_items = predict_similar_items(prediction_graph_factory=self.prediction_graph_factory,
                                                              tf_item_representation=self.tf_item_representation,
                                                              tf_similar_items_ids=self.tf_similar_items_ids)
        self.tf_rank_similar_items = rank_predictions(tf_prediction=self.tf_predict_similar_items)

        # Compose loss function args
        # This composition is for execution safety: it prevents loss functions that are incorrectly configured from
        # having visibility of certain nodes.
        loss_graph_kwargs = {
            'tf_prediction_serial': self.tf_prediction_serial,
            'tf_interactions_serial': tf_interactions_serial,
            'tf_interactions': tf_interactions,
            'tf_n_users': tf_n_users,
            'tf_n_items': tf_n_items,
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
                tf_n_users=tf_n_users,
            )
            loss_graph_kwargs.update({'tf_sample_predictions': tf_sample_predictions,
                                      'tf_n_sampled_items': self.tf_n_sampled_items})

        # Build loss graph
        self.tf_basic_loss = self.loss_graph_factory.connect_loss_graph(**loss_graph_kwargs)

        self.tf_weight_reg_loss = sum(tf.nn.l2_loss(weights) for weights in tf_weights)
        self.tf_loss = self.tf_basic_loss + (self.tf_alpha * self.tf_weight_reg_loss)
        self.tf_optimizer = tf.train.AdamOptimizer(learning_rate=self.tf_learning_rate).minimize(self.tf_loss)

        # Record the new node names
        self._record_graph_hook_names()

    def fit(self, interactions, user_features, item_features, epochs=100, learning_rate=0.1, alpha=0.00001,
            verbose=False, user_batch_size=None, n_sampled_items=None):
        """
        Constructs the TensorRec graph and fits the model.
        :param interactions: scipy.sparse matrix, tensorflow.data.Dataset, str, or list
        A matrix of interactions of shape [n_users, n_items].
        If a Dataset, the Dataset must follow the format used in tensorrec.input_utils.
        If a str, the string must be the path to a TFRecord file.
        If a list, the list must contain scipy.sparse matrices, tensorflow.data.Datasets, or strs.
        :param user_features: scipy.sparse matrix, tensorflow.data.Dataset, str, or list
        A matrix of user features of shape [n_users, n_user_features].
        If a Dataset, the Dataset must follow the format used in tensorrec.input_utils.
        If a str, the string must be the path to a TFRecord file.
        If a list, the list must contain scipy.sparse matrices, tensorflow.data.Datasets, or strs.
        :param item_features: scipy.sparse matrix, tensorflow.data.Dataset, str, or list
        A matrix of item features of shape [n_items, n_item_features].
        If a Dataset, the Dataset must follow the format used in tensorrec.input_utils.
        If a str, the string must be the path to a TFRecord file.
        If a list, the list must contain scipy.sparse matrices, tensorflow.data.Datasets, or strs.
        :param epochs: Integer
        The number of epochs to fit the model.
        :param learning_rate: Float
        The learning rate of the model.
        :param alpha:
        The weight regularization loss coefficient.
        :param verbose: boolean
        If true, the model will print a number of status statements during fitting.
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
                         user_batch_size=user_batch_size,
                         n_sampled_items=n_sampled_items)

    def fit_partial(self, interactions, user_features, item_features, epochs=1, learning_rate=0.1,
                    alpha=0.00001, verbose=False, user_batch_size=None, n_sampled_items=None):
        """
        Constructs the TensorRec graph and fits the model.
        :param interactions: scipy.sparse matrix, tensorflow.data.Dataset, str, or list
        A matrix of interactions of shape [n_users, n_items].
        If a Dataset, the Dataset must follow the format used in tensorrec.input_utils.
        If a str, the string must be the path to a TFRecord file.
        If a list, the list must contain scipy.sparse matrices, tensorflow.data.Datasets, or strs.
        :param user_features: scipy.sparse matrix, tensorflow.data.Dataset, str, or list
        A matrix of user features of shape [n_users, n_user_features].
        If a Dataset, the Dataset must follow the format used in tensorrec.input_utils.
        If a str, the string must be the path to a TFRecord file.
        If a list, the list must contain scipy.sparse matrices, tensorflow.data.Datasets, or strs.
        :param item_features: scipy.sparse matrix, tensorflow.data.Dataset, str, or list
        A matrix of item features of shape [n_items, n_item_features].
        If a Dataset, the Dataset must follow the format used in tensorrec.input_utils.
        If a str, the string must be the path to a TFRecord file.
        If a list, the list must contain scipy.sparse matrices, tensorflow.data.Datasets, or strs.
        :param epochs: Integer
        The number of epochs to fit the model.
        :param learning_rate: Float
        The learning rate of the model.
        :param alpha:
        The weight regularization loss coefficient.
        :param verbose: boolean
        If true, the model will print a number of status statements during fitting.
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

        # Check if the iterators have been constructed. If not, build them.
        if self.tf_interaction_iterator is None:
            self._build_input_iterators()

        if verbose:
            logging.info('Processing interaction and feature data')

        dataset_sets, initializer_sets = self._create_batched_dataset_initializers(interactions=interactions,
                                                                                   user_features=user_features,
                                                                                   item_features=item_features,
                                                                                   user_batch_size=user_batch_size)

        # Check if the graph has been constructed by checking the dense prediction node
        # If it hasn't been constructed, initialize it
        if self.tf_prediction is None:

            # Check input dimensions
            first_batch = dataset_sets[0]
            _, n_user_features = get_dimensions_from_tensorrec_dataset(first_batch[1])
            _, n_item_features = get_dimensions_from_tensorrec_dataset(first_batch[2])

            # Numbers of features are either learned at fit time from the shape of these two matrices or specified at
            # TensorRec construction and cannot be changed.
            self._build_tf_graph(n_user_features=n_user_features, n_item_features=n_item_features)
            session.run(tf.global_variables_initializer())

        # Build the shared feed dict
        feed_dict = {self.tf_learning_rate: learning_rate,
                     self.tf_alpha: calculate_batched_alpha(num_batches=len(initializer_sets), alpha=alpha)}
        if self.loss_graph_factory.is_sample_based:
            feed_dict[self.tf_n_sampled_items] = n_sampled_items

        if verbose:
            logging.info('Beginning fitting')

        for epoch in range(epochs):
            for batch, initializers in enumerate(initializer_sets):

                session.run(initializers)
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

    def predict(self, user_features, item_features):
        """
        Predict recommendation scores for the given users and items.
        :param user_features: scipy.sparse matrix, tensorflow.data.Dataset, str, or list
        A matrix of user features of shape [n_users, n_user_features].
        If a Dataset, the Dataset must follow the format used in tensorrec.input_utils.
        If a str, the string must be the path to a TFRecord file.
        If a list, the list must contain scipy.sparse matrices, tensorflow.data.Datasets, or strs.
        :param item_features: scipy.sparse matrix, tensorflow.data.Dataset, str, or list
        A matrix of item features of shape [n_items, n_item_features].
        If a Dataset, the Dataset must follow the format used in tensorrec.input_utils.
        If a str, the string must be the path to a TFRecord file.
        If a list, the list must contain scipy.sparse matrices, tensorflow.data.Datasets, or strs.
        :return: np.ndarray
        The predictions in an ndarray of shape [n_users, n_items]
        """

        # Ensure that the model has been fit
        if self.tf_prediction is None:
            raise ModelNotFitException(method='predict')

        _, initializers = self._create_datasets_and_initializers(interactions=None,
                                                                 user_features=user_features,
                                                                 item_features=item_features)
        get_session().run(initializers)

        predictions = self.tf_prediction.eval(session=get_session())

        return predictions

    def predict_similar_items(self, item_features, item_ids, n_similar):
        """
        Predicts the most similar items to the given item_ids.
        :param item_features: scipy.sparse matrix, tensorflow.data.Dataset, str, or list
        A matrix of item features of shape [n_items, n_item_features].
        If a Dataset, the Dataset must follow the format used in tensorrec.input_utils.
        If a str, the string must be the path to a TFRecord file.
        If a list, the list must contain scipy.sparse matrices, tensorflow.data.Datasets, or strs.
        :param item_ids: list or np.array
        The ids of the items of interest.
        E.g. [4, 8, 12] to get sims for items 4, 8, and 12.
        :param n_similar: int
        The number of similar items to get per item of interest.
        :return: list of lists of tuples
        The first level list corresponds to input arg item_ids.
        The second level list is of length n_similar and contains tuples of (item_id, score) for each similar item.
        """

        # Ensure that the model has been fit
        if self.tf_prediction is None:
            raise ModelNotFitException(method='predict_similar_items')

        _, initializers = self._create_datasets_and_initializers(interactions=None,
                                                                 user_features=None,
                                                                 item_features=item_features)
        get_session().run(initializers)

        feed_dict = {self.tf_similar_items_ids: np.array(item_ids)}
        sims = self.tf_predict_similar_items.eval(session=get_session(), feed_dict=feed_dict)

        results = []
        for i in range(len(item_ids)):
            item_sims = sims[i]
            best = np.argpartition(item_sims, -n_similar)[-n_similar:]
            item_results = sorted(zip(best, item_sims[best]), key=lambda x: -x[1])
            results.append(item_results)

        return results

    def predict_rank(self, user_features, item_features):
        """
        Predict recommendation ranks for the given users and items.
        :param user_features: scipy.sparse matrix, tensorflow.data.Dataset, str, or list
        A matrix of user features of shape [n_users, n_user_features].
        If a Dataset, the Dataset must follow the format used in tensorrec.input_utils.
        If a str, the string must be the path to a TFRecord file.
        If a list, the list must contain scipy.sparse matrices, tensorflow.data.Datasets, or strs.
        :param item_features: scipy.sparse matrix, tensorflow.data.Dataset, str, or list
        A matrix of item features of shape [n_items, n_item_features].
        If a Dataset, the Dataset must follow the format used in tensorrec.input_utils.
        If a str, the string must be the path to a TFRecord file.
        If a list, the list must contain scipy.sparse matrices, tensorflow.data.Datasets, or strs.
        :return: np.ndarray
        The ranks in an ndarray of shape [n_users, n_items]
        """

        # Ensure that the model has been fit
        if self.tf_prediction is None:
            raise ModelNotFitException(method='predict_rank')

        _, initializers = self._create_datasets_and_initializers(interactions=None,
                                                                 user_features=user_features,
                                                                 item_features=item_features)
        get_session().run(initializers)

        rankings = self.tf_rankings.eval(session=get_session())

        return rankings

    def predict_user_representation(self, user_features):
        """
        Predict latent representation vectors for the given users.
        :param user_features: scipy.sparse matrix, tensorflow.data.Dataset, str, or list
        A matrix of user features of shape [n_users, n_user_features].
        If a Dataset, the Dataset must follow the format used in tensorrec.input_utils.
        If a str, the string must be the path to a TFRecord file.
        If a list, the list must contain scipy.sparse matrices, tensorflow.data.Datasets, or strs.
        :return: np.ndarray
        The latent user representations in an ndarray of shape [n_users, n_components]
        """

        # Ensure that the model has been fit
        if self.tf_prediction is None:
            raise ModelNotFitException(method='predict_user_representation')

        _, initializers = self._create_datasets_and_initializers(interactions=None,
                                                                 user_features=user_features,
                                                                 item_features=None)
        get_session().run(initializers)

        user_repr = self.tf_user_representation.eval(session=get_session())

        # If there is only one user repr per user, collapse from rank 3 to rank 2
        if self.n_tastes == 1:
            user_repr = np.sum(user_repr, axis=0)

        return user_repr

    def predict_user_attention_representation(self, user_features):
        """
        Predict latent attention representation vectors for the given users.
        :param user_features: scipy.sparse matrix, tensorflow.data.Dataset, str, or list
        A matrix of user features of shape [n_users, n_user_features].
        If a Dataset, the Dataset must follow the format used in tensorrec.input_utils.
        If a str, the string must be the path to a TFRecord file.
        If a list, the list must contain scipy.sparse matrices, tensorflow.data.Datasets, or strs.
        :return: np.ndarray
        The latent user attention representations in an ndarray of shape [n_users, n_components]
        """

        # Ensure that the model has been fit
        if self.tf_prediction is None:
            raise ModelNotFitException(method='predict_user_attention_representation')

        if self.attention_graph_factory is None:
            raise ModelWithoutAttentionException()

        _, initializers = self._create_datasets_and_initializers(interactions=None,
                                                                 user_features=user_features,
                                                                 item_features=None)
        get_session().run(initializers)
        user_attn_repr = self.tf_user_attention_representation.eval(session=get_session())

        # If there is only one user attn repr per user, collapse from rank 3 to rank 2
        if self.n_tastes == 1:
            user_attn_repr = np.sum(user_attn_repr, axis=0)

        return user_attn_repr

    def predict_item_representation(self, item_features):
        """
        Predict representation vectors for the given items.
        :param item_features: scipy.sparse matrix, tensorflow.data.Dataset, str, or list
        A matrix of item features of shape [n_items, n_item_features].
        If a Dataset, the Dataset must follow the format used in tensorrec.input_utils.
        If a str, the string must be the path to a TFRecord file.
        If a list, the list must contain scipy.sparse matrices, tensorflow.data.Datasets, or strs.
        :return: np.ndarray
        The latent item representations in an ndarray of shape [n_items, n_components]
        """

        # Ensure that the model has been fit
        if self.tf_prediction is None:
            raise ModelNotFitException(method='predict_item_representation')

        _, initializers = self._create_datasets_and_initializers(interactions=None,
                                                                 user_features=None,
                                                                 item_features=item_features)
        get_session().run(initializers)
        item_repr = self.tf_item_representation.eval(session=get_session())
        return item_repr

    def predict_user_bias(self, user_features):
        """
        Predict bias values for the given users.
        :param user_features: scipy.sparse matrix, tensorflow.data.Dataset, str, or list
        A matrix of user features of shape [n_users, n_user_features].
        If a Dataset, the Dataset must follow the format used in tensorrec.input_utils.
        If a str, the string must be the path to a TFRecord file.
        If a list, the list must contain scipy.sparse matrices, tensorflow.data.Datasets, or strs.
        :return: np.ndarray
        The user biases in an ndarray of shape [n_users]
        """

        # Ensure that the model has been fit
        if self.tf_prediction is None:
            raise ModelNotFitException(method='predict_user_bias')

        if not self.biased:
            raise ModelNotBiasedException(actor='user')

        _, initializers = self._create_datasets_and_initializers(interactions=None,
                                                                 user_features=user_features,
                                                                 item_features=None)
        get_session().run(initializers)
        predictions = self.tf_projected_user_biases.eval(session=get_session())
        return predictions

    def predict_item_bias(self, item_features):
        """
        Predict bias values for the given items.
        :param item_features: scipy.sparse matrix, tensorflow.data.Dataset, str, or list
        A matrix of item features of shape [n_items, n_item_features].
        If a Dataset, the Dataset must follow the format used in tensorrec.input_utils.
        If a str, the string must be the path to a TFRecord file.
        If a list, the list must contain scipy.sparse matrices, tensorflow.data.Datasets, or strs.
        :return: np.ndarray
        The item biases in an ndarray of shape [n_items]
        """

        # Ensure that the model has been fit
        if self.tf_prediction is None:
            raise ModelNotFitException(method='predict_item_bias')

        if not self.biased:
            raise ModelNotBiasedException(actor='item')

        _, initializers = self._create_datasets_and_initializers(interactions=None,
                                                                 user_features=None,
                                                                 item_features=item_features)
        get_session().run(initializers)
        predictions = self.tf_projected_item_biases.eval(session=get_session())
        return predictions

    def save_model(self, directory_path):
        """
        Saves the model to files in the given directory.
        :param directory_path: str
        The path to the directory in which to save the model.
        :return:
        """

        # Ensure that the model has been fit
        if self.tf_prediction is None:
            raise ModelNotFitException(method='save_model')

        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        saver = tf.train.Saver()
        session_path = os.path.join(directory_path, 'tensorrec_session.cpkt')
        saver.save(sess=get_session(), save_path=session_path)

        # Break connections to the graph before saving the python object
        self._break_graph_hooks()
        tensorrec_path = os.path.join(directory_path, 'tensorrec.pkl')
        with open(tensorrec_path, 'wb') as file:
            pickle.dump(file=file, obj=self)

        # Reconnect to the graph after saving
        self._attach_graph_hooks()

    @classmethod
    def load_model(cls, directory_path):
        """
        Loads the TensorRec model and TensorFlow session saved in the given directory.
        :param directory_path: str
        The path to the directory containing the saved model.
        :return:
        """

        graph_path = os.path.join(directory_path, 'tensorrec_session.cpkt.meta')
        saver = tf.train.import_meta_graph(graph_path)

        session_path = os.path.join(directory_path, 'tensorrec_session.cpkt')
        saver.restore(sess=get_session(), save_path=session_path)

        tensorrec_path = os.path.join(directory_path, 'tensorrec.pkl')
        with open(tensorrec_path, 'rb') as file:
            model = pickle.load(file=file)
        model._attach_graph_hooks()
        return model
