from unittest import TestCase

import numpy as np
import tensorflow as tf

from tensorrec import TensorRec
from tensorrec.eval import recall_at_k
from tensorrec.util import generate_dummy_data


class TensorRecTestCase(TestCase):

    def test_init(self):
        self.assertIsNotNone(TensorRec())

    def test_init_fail_0_components(self):
        with self.assertRaises(ValueError):
            TensorRec(n_components=0)

    def test_init_fail_none_factory(self):
        with self.assertRaises(ValueError):
            TensorRec(user_repr_graph_factory=None)
        with self.assertRaises(ValueError):
            TensorRec(item_repr_graph_factory=None)
        with self.assertRaises(ValueError):
            TensorRec(loss_graph_factory=None)

    def test_fit(self):
        interactions, user_features, item_features = generate_dummy_data(num_users=10,
                                                                         num_items=10,
                                                                         interaction_density=.5)
        model = TensorRec(n_components=10)
        session = tf.Session()
        model.fit(session, interactions, user_features, item_features, epochs=10)
        # Ensure that the nodes have been built
        self.assertIsNotNone(model.tf_prediction_dense)

    def test_predict(self):
        interactions, user_features, item_features = generate_dummy_data(num_users=10,
                                                                         num_items=10,
                                                                         interaction_density=.5)
        model = TensorRec(n_components=10)
        session = tf.Session()
        model.fit(session, interactions, user_features, item_features, epochs=10)

        predictions = model.predict(session,
                                    user_ids=[1, 2, 3],
                                    item_ids=[4, 5, 6],
                                    user_features=user_features,
                                    item_features=item_features)

        self.assertEqual(len(predictions), 3)


class ReadmeTestCase(TestCase):

    def test_basic_usage(self):
        # Build the model with default parameters
        model = TensorRec()

        # Generate some dummy data
        interactions, user_features, item_features = generate_dummy_data(num_users=100,
                                                                         num_items=150,
                                                                         interaction_density=.05)

        # Start a TensorFlow session and fit the model
        session = tf.Session()
        model.fit(session, interactions, user_features, item_features, epochs=5, verbose=True)

        # Predict scores for user 75 on items 100, 101, and 102
        predictions = model.predict(session,
                                    user_ids=[75, 75, 75],
                                    item_ids=[100, 101, 102],
                                    user_features=user_features,
                                    item_features=item_features)

        # Calculate and print the recall at 10
        r_at_k = recall_at_k(model, session, interactions,
                             k=10,
                             user_features=user_features,
                             item_features=item_features)
        print(np.mean(r_at_k))

        self.assertIsNotNone(predictions)

    def test_custom_repr_graph(self):
        # Define a custom representation function graph
        def build_tanh_representation_graph(tf_features, n_components, n_features, node_name_ending):
            tf_tanh_weights = tf.Variable(tf.random_normal([n_features, n_components],
                                                           stddev=.5),
                                          name='tanh_weights_%s' % node_name_ending)

            tf_repr = tf.nn.tanh(tf.sparse_tensor_dense_matmul(tf_features, tf_tanh_weights))

            # Return repr layer and variables
            return tf_repr, [tf_tanh_weights]

        # Build a model with the custom representation function
        model = TensorRec(user_repr_graph_factory=build_tanh_representation_graph,
                          item_repr_graph_factory=build_tanh_representation_graph)

        self.assertIsNotNone(model)

    def test_custom_loss_graph(self):
        # Define a custom loss function graph
        def build_simple_error_graph(tf_prediction, tf_y, **kwargs):
            return tf.reduce_mean(tf.abs(tf_y - tf_prediction))

        # Build a model with the custom loss function
        model = TensorRec(loss_graph_factory=build_simple_error_graph)

        self.assertIsNotNone(model)
