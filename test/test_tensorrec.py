import shutil
import tempfile
from unittest import TestCase

import numpy as np
import tensorflow as tf

from tensorrec import TensorRec
from tensorrec.eval import recall_at_k
from tensorrec.util import generate_dummy_data_with_indicator, generate_dummy_data
from tensorrec.session_management import set_session


class TensorRecTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.interactions, cls.user_features, cls.item_features = generate_dummy_data(
            num_users=15, num_items=30, interaction_density=.5, num_user_features=200, num_item_features=200,
            n_features_per_user=20, n_features_per_item=20, pos_int_ratio=.5
        )

    def test_init(self):
        self.assertIsNotNone(TensorRec())

    def test_init_fail_0_components(self):
        with self.assertRaises(ValueError):
            TensorRec(n_components=0)

    def test_init_fail_none_factory(self):
        with self.assertRaises(ValueError):
            TensorRec(user_repr_graph=None)
        with self.assertRaises(ValueError):
            TensorRec(item_repr_graph=None)
        with self.assertRaises(ValueError):
            TensorRec(loss_graph=None)

    def test_fit(self):
        model = TensorRec(n_components=10)
        model.fit(self.interactions, self.user_features, self.item_features, epochs=10)
        # Ensure that the nodes have been built
        self.assertIsNotNone(model.tf_prediction)

    def test_predict(self):
        model = TensorRec(n_components=10)
        model.fit(self.interactions, self.user_features, self.item_features, epochs=10)

        predictions = model.predict(user_features=self.user_features,
                                    item_features=self.item_features)

        self.assertEqual(predictions.shape, (self.user_features.shape[0], self.item_features.shape[0]))

    def test_fit_predict_unbiased(self):
        model = TensorRec(n_components=10, biased=False)
        model.fit(self.interactions, self.user_features, self.item_features, epochs=10)

        predictions = model.predict(user_features=self.user_features,
                                    item_features=self.item_features)

        self.assertEqual(predictions.shape, (self.user_features.shape[0], self.item_features.shape[0]))

    def test_predict_user_repr(self):
        model = TensorRec(n_components=10, biased=False)
        model.fit(self.interactions, self.user_features, self.item_features, epochs=10)

        user_repr = model.predict_user_representation(self.user_features)
        self.assertEqual(user_repr.shape, (self.user_features.shape[0], 10))

    def test_predict_item_repr(self):
        model = TensorRec(n_components=10, biased=False)
        model.fit(self.interactions, self.user_features, self.item_features, epochs=10)

        item_repr = model.predict_item_representation(self.item_features)
        self.assertEqual(item_repr.shape, (self.item_features.shape[0], 10))

    def test_predict_user_repr_biased_fails(self):
        model = TensorRec(n_components=10)
        model.fit(self.interactions, self.user_features, self.item_features, epochs=10)

        with self.assertRaises(NotImplementedError):
            model.predict_user_representation(self.user_features)

    def test_predict_item_repr_biased_fails(self):
        model = TensorRec(n_components=10)
        model.fit(self.interactions, self.user_features, self.item_features, epochs=10)

        with self.assertRaises(NotImplementedError):
            model.predict_item_representation(self.item_features)


class TensorRecWithIndicatorTestCase(TensorRecTestCase):

    @classmethod
    def setUpClass(cls):
        cls.interactions, cls.user_features, cls.item_features = generate_dummy_data_with_indicator(
            num_users=10, num_items=20, interaction_density=.5
        )


class TensorRecSavingTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.interactions, cls.user_features, cls.item_features = generate_dummy_data(
            num_users=15, num_items=30, interaction_density=.5, num_user_features=200, num_item_features=200,
            n_features_per_user=20, n_features_per_item=20, pos_int_ratio=.5
        )

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_save_and_load_model(self):
        model = TensorRec(n_components=10)
        model.fit(self.interactions, self.user_features, self.item_features, epochs=10)

        predictions = model.predict(user_features=self.user_features, item_features=self.item_features)
        ranks = model.predict_rank(user_features=self.user_features, item_features=self.item_features)
        model.save_model(directory_path=self.test_dir)

        # Blow away the session
        set_session(tf.Session())

        # Reload the model, predict, and check for equal predictions
        new_model = TensorRec.load_model(directory_path=self.test_dir)
        new_predictions = new_model.predict(user_features=self.user_features, item_features=self.item_features)
        new_ranks = new_model.predict_rank(user_features=self.user_features, item_features=self.item_features)

        self.assertEqual(predictions.all(), new_predictions.all())
        self.assertEqual(ranks.all(), new_ranks.all())

    def test_save_and_load_model_same_session(self):
        model = TensorRec(n_components=10)
        model.fit(self.interactions, self.user_features, self.item_features, epochs=10)

        predictions = model.predict(user_features=self.user_features, item_features=self.item_features)
        ranks = model.predict_rank(user_features=self.user_features, item_features=self.item_features)
        model.save_model(directory_path=self.test_dir)

        # Reload the model, predict, and check for equal predictions
        new_model = TensorRec.load_model(directory_path=self.test_dir)
        new_predictions = new_model.predict(user_features=self.user_features, item_features=self.item_features)
        new_ranks = new_model.predict_rank(user_features=self.user_features, item_features=self.item_features)

        self.assertEqual(predictions.all(), new_predictions.all())
        self.assertEqual(ranks.all(), new_ranks.all())


class ReadmeTestCase(TestCase):

    def test_basic_usage(self):
        # Build the model with default parameters
        model = TensorRec()

        # Generate some dummy data
        interactions, user_features, item_features = generate_dummy_data(num_users=100,
                                                                         num_items=150,
                                                                         interaction_density=.05)

        # Fit the model
        model.fit(interactions, user_features, item_features, epochs=5, verbose=True)

        # Predict scores for all users and all items
        predictions = model.predict(user_features=user_features,
                                    item_features=item_features)

        # Calculate and print the recall at 10
        r_at_k = recall_at_k(model, interactions,
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
        model = TensorRec(user_repr_graph=build_tanh_representation_graph,
                          item_repr_graph=build_tanh_representation_graph)

        self.assertIsNotNone(model)

    def test_custom_loss_graph(self):
        # Define a custom loss function graph
        def build_simple_error_graph(tf_prediction, tf_y, **kwargs):
            return tf.reduce_mean(tf.abs(tf_y - tf_prediction))

        # Build a model with the custom loss function
        model = TensorRec(loss_graph=build_simple_error_graph)

        self.assertIsNotNone(model)
