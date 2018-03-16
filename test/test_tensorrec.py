import numpy as np
import shutil
import tempfile
from unittest import TestCase

import tensorflow as tf

from tensorrec import TensorRec
from tensorrec.util import generate_dummy_data_with_indicator, generate_dummy_data
from tensorrec.session_management import set_session


class TensorRecTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.interactions, cls.user_features, cls.item_features = generate_dummy_data(
            num_users=15, num_items=30, interaction_density=.5, num_user_features=200, num_item_features=200,
            n_features_per_user=20, n_features_per_item=20, pos_int_ratio=.5
        )

        cls.standard_model = TensorRec(n_components=10)
        cls.standard_model.fit(cls.interactions, cls.user_features, cls.item_features, epochs=10)

        cls.unbiased_model = TensorRec(n_components=10, biased=False)
        cls.unbiased_model.fit(cls.interactions, cls.user_features, cls.item_features, epochs=10)

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

    def test_init_fail_bad_loss_graph(self):
        with self.assertRaises(ValueError):
            TensorRec(loss_graph=np.mean)

    def test_fit(self):
        # Ensure that the nodes have been built
        self.assertIsNotNone(self.standard_model.tf_prediction)

    def test_fit_verbose(self):
        model = TensorRec(n_components=10)
        model.fit(self.interactions, self.user_features, self.item_features, epochs=10, verbose=True)
        # Ensure that the nodes have been built
        self.assertIsNotNone(model.tf_prediction)

    def test_fit_batched(self):
        model = TensorRec(n_components=10)
        model.fit(self.interactions, self.user_features, self.item_features, epochs=10, user_batch_size=2)
        # Ensure that the nodes have been built
        self.assertIsNotNone(model.tf_prediction)

    def test_predict(self):
        predictions = self.standard_model.predict(user_features=self.user_features,
                                                  item_features=self.item_features)

        self.assertEqual(predictions.shape, (self.user_features.shape[0], self.item_features.shape[0]))

    def test_predict_dot_product(self):
        predictions = self.standard_model.predict_dot_product(user_features=self.user_features,
                                                              item_features=self.item_features)

        self.assertEqual(predictions.shape, (self.user_features.shape[0], self.item_features.shape[0]))

    def test_predict_cosine_similarity(self):
        cosines = self.standard_model.predict_cosine_similarity(user_features=self.user_features,
                                                                item_features=self.item_features)

        self.assertEqual(cosines.shape, (self.user_features.shape[0], self.item_features.shape[0]))
        for x in range(cosines.shape[0]):
            for y in range(cosines.shape[1]):
                val = cosines[x][y]
                self.assertGreaterEqual(val, -1.0)
                self.assertLessEqual(val, 1.0)

    def test_predict_euclidian_similarity(self):
        cosines = self.standard_model.predict_euclidian_similarity(user_features=self.user_features,
                                                                   item_features=self.item_features)

        self.assertEqual(cosines.shape, (self.user_features.shape[0], self.item_features.shape[0]))
        for x in range(cosines.shape[0]):
            for y in range(cosines.shape[1]):
                val = cosines[x][y]
                self.assertLessEqual(val, 0.0)

    def test_predict_rank(self):
        ranks = self.standard_model.predict_rank(user_features=self.user_features,
                                                 item_features=self.item_features)

        self.assertEqual(ranks.shape, (self.user_features.shape[0], self.item_features.shape[0]))
        for x in range(ranks.shape[0]):
            for y in range(ranks.shape[1]):
                val = ranks[x][y]
                self.assertGreater(val, 0)

    def test_fit_predict_unbiased(self):
        predictions = self.unbiased_model.predict(user_features=self.user_features, item_features=self.item_features)
        self.assertEqual(predictions.shape, (self.user_features.shape[0], self.item_features.shape[0]))

    def test_predict_user_repr(self):
        user_repr = self.unbiased_model.predict_user_representation(self.user_features)
        self.assertEqual(user_repr.shape, (self.user_features.shape[0], 10))

    def test_predict_item_repr(self):
        item_repr = self.unbiased_model.predict_item_representation(self.item_features)
        self.assertEqual(item_repr.shape, (self.item_features.shape[0], 10))

    def test_predict_user_bias_unbiased_model(self):
        self.assertRaises(
            NotImplementedError,
            self.unbiased_model.predict_user_bias,
            self.user_features)

    def test_predict_item_bias_unbiased_model(self):
        self.assertRaises(
            NotImplementedError,
            self.unbiased_model.predict_item_bias,
            self.item_features)


class TensorRecBiasedPrediction(TestCase):
    # TODO: Collapse these into TensorRecTestCase once the fit bug is fixed

    @classmethod
    def setUpClass(cls):
        cls.interactions, cls.user_features, cls.item_features = generate_dummy_data(
            num_users=15, num_items=30, interaction_density=.5, num_user_features=200, num_item_features=200,
            n_features_per_user=20, n_features_per_item=20, pos_int_ratio=.5
        )

        cls.standard_model = TensorRec(n_components=10)
        cls.standard_model.fit(cls.interactions, cls.user_features, cls.item_features, epochs=10)

    def test_predict_user_bias(self):
        user_bias = self.standard_model.predict_user_bias(self.user_features)
        self.assertTrue(any(user_bias))  # Make sure it isn't all 0s

    def test_predict_item_bias(self):
        item_bias = self.standard_model.predict_item_bias(self.item_features)
        self.assertTrue(any(item_bias))  # Make sure it isn't all 0s


class TensorRecNormalizedTestCase(TensorRecTestCase):

    @classmethod
    def setUpClass(cls):
        cls.interactions, cls.user_features, cls.item_features = generate_dummy_data_with_indicator(
            num_users=10, num_items=20, interaction_density=.5
        )

        cls.standard_model = TensorRec(n_components=10, normalize_items=True, normalize_users=True)
        cls.standard_model.fit(cls.interactions, cls.user_features, cls.item_features, epochs=10)

        cls.unbiased_model = TensorRec(n_components=10, normalize_items=True, normalize_users=True, biased=False)
        cls.unbiased_model.fit(cls.interactions, cls.user_features, cls.item_features, epochs=10)


class TensorRecNTastesTestCase(TensorRecTestCase):

    @classmethod
    def setUpClass(cls):
        cls.interactions, cls.user_features, cls.item_features = generate_dummy_data(
            num_users=15, num_items=30, interaction_density=.5, num_user_features=200, num_item_features=200,
            n_features_per_user=20, n_features_per_item=20, pos_int_ratio=.5
        )

        cls.standard_model = TensorRec(n_components=10, n_tastes=3)
        cls.standard_model.fit(cls.interactions, cls.user_features, cls.item_features, epochs=10)

        cls.unbiased_model = TensorRec(n_components=10, n_tastes=3, biased=False)
        cls.unbiased_model.fit(cls.interactions, cls.user_features, cls.item_features, epochs=10)

    def test_predict_user_repr(self):
        user_repr = self.unbiased_model.predict_user_representation(self.user_features)

        # 3 tastes, shape[0] users, 10 components
        self.assertEqual(user_repr.shape, (3, self.user_features.shape[0], 10))


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

        # Check that, after saving, the same predictions come back
        predictions_after_save = model.predict(user_features=self.user_features, item_features=self.item_features)
        ranks_after_save = model.predict_rank(user_features=self.user_features, item_features=self.item_features)
        self.assertTrue((predictions == predictions_after_save).all())
        self.assertTrue((ranks == ranks_after_save).all())

        # Blow away the session
        set_session(tf.Session())

        # Reload the model, predict, and check for equal predictions
        new_model = TensorRec.load_model(directory_path=self.test_dir)
        new_predictions = new_model.predict(user_features=self.user_features, item_features=self.item_features)
        new_ranks = new_model.predict_rank(user_features=self.user_features, item_features=self.item_features)

        self.assertTrue((predictions == new_predictions).all())
        self.assertTrue((ranks == new_ranks).all())

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

        self.assertTrue((predictions == new_predictions).all())
        self.assertTrue((ranks == new_ranks).all())
