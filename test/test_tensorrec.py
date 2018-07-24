import numpy as np
import os
import shutil
import tempfile
from unittest import TestCase

import tensorflow as tf

from tensorrec import TensorRec
from tensorrec.errors import (
    ModelNotBiasedException, ModelNotFitException, ModelWithoutAttentionException, BatchNonSparseInputException
)
from tensorrec.input_utils import create_tensorrec_dataset_from_sparse_matrix, write_tfrecord_from_sparse_matrix
from tensorrec.representation_graphs import NormalizedLinearRepresentationGraph, LinearRepresentationGraph
from tensorrec.session_management import set_session
from tensorrec.util import generate_dummy_data


class TensorRecTestCase(TestCase):

    @classmethod
    def setUpClass(cls):

        cls.n_user_features = 200
        cls.n_item_features = 150

        cls.interactions, cls.user_features, cls.item_features = generate_dummy_data(
            num_users=15, num_items=30, interaction_density=.5, num_user_features=cls.n_user_features,
            num_item_features=cls.n_item_features, n_features_per_user=20, n_features_per_item=20, pos_int_ratio=.5
        )

        set_session(None)
        cls.temp_dir = tempfile.mkdtemp()
        cls.interactions_path = os.path.join(cls.temp_dir, 'interactions.tfrecord')
        cls.user_features_path = os.path.join(cls.temp_dir, 'user_features.tfrecord')
        cls.item_features_path = os.path.join(cls.temp_dir, 'item_features.tfrecord')

        write_tfrecord_from_sparse_matrix(cls.user_features_path, cls.user_features)
        write_tfrecord_from_sparse_matrix(cls.item_features_path, cls.item_features)
        write_tfrecord_from_sparse_matrix(cls.interactions_path, cls.interactions)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

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

    def test_init_fail_attention_with_1_taste(self):
        with self.assertRaises(ValueError):
            TensorRec(n_tastes=1, attention_graph=LinearRepresentationGraph())

    def test_init_fail_bad_attention_graph(self):
        with self.assertRaises(ValueError):
            TensorRec(attention_graph=np.mean)

    def test_predict_fail_unfit(self):
        model = TensorRec()
        with self.assertRaises(ModelNotFitException):
            model.predict(self.user_features, self.item_features)
        with self.assertRaises(ModelNotFitException):
            model.predict_rank(self.user_features, self.item_features)

        with self.assertRaises(ModelNotFitException):
            model.predict_user_representation(self.user_features)
        with self.assertRaises(ModelNotFitException):
            model.predict_item_representation(self.item_features)
        with self.assertRaises(ModelNotFitException):
            model.predict_user_attention_representation(self.user_features)

        with self.assertRaises(ModelNotFitException):
            model.predict_similar_items(self.item_features, item_ids=[1], n_similar=5)

        with self.assertRaises(ModelNotFitException):
            model.predict_item_bias(self.item_features)
        with self.assertRaises(ModelNotFitException):
            model.predict_user_bias(self.user_features)

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

    def test_fit_fail_bad_input(self):
        model = TensorRec(n_components=10)
        with self.assertRaises(ValueError):
            model.fit(np.array([1, 2, 3, 4]), self.user_features, self.item_features, epochs=10)
        with self.assertRaises(ValueError):
            model.fit(self.interactions, np.array([1, 2, 3, 4]), self.item_features, epochs=10)
        with self.assertRaises(ValueError):
            model.fit(self.interactions, self.user_features, np.array([1, 2, 3, 4]), epochs=10)

    def test_fit_fail_mismatched_batches(self):
        model = TensorRec(n_components=10)
        with self.assertRaises(ValueError):
            model.fit(self.interactions,
                      [self.user_features, self.user_features],
                      [self.item_features, self.item_features, self.item_features],
                      epochs=10)

        with self.assertRaises(ValueError):
            model.fit(self.interactions,
                      [self.user_features, self.user_features],
                      [self.item_features, self.item_features],
                      epochs=10)

        model.fit([self.interactions, self.interactions],
                  [self.user_features, self.user_features],
                  self.item_features,
                  epochs=10)

        model.fit([self.interactions, self.interactions],
                  [self.user_features, self.user_features],
                  [self.item_features, self.item_features],
                  epochs=10)

    def test_fit_fail_batching_dataset(self):
        model = TensorRec(n_components=10)

        interactions_as_dataset = create_tensorrec_dataset_from_sparse_matrix(self.interactions)
        with self.assertRaises(BatchNonSparseInputException):
            model.fit(interactions_as_dataset, self.user_features, self.item_features, epochs=10, user_batch_size=2)

    def test_fit_user_feature_as_dataset(self):
        uf_as_dataset = create_tensorrec_dataset_from_sparse_matrix(self.user_features)
        model = TensorRec(n_components=10)
        model.fit(self.interactions, uf_as_dataset, self.item_features, epochs=10)

    def test_fit_item_feature_as_dataset(self):
        if_as_dataset = create_tensorrec_dataset_from_sparse_matrix(self.item_features)
        model = TensorRec(n_components=10)
        model.fit(self.interactions, self.user_features, if_as_dataset, epochs=10)

    def test_fit_interactions_as_dataset(self):
        int_as_dataset = create_tensorrec_dataset_from_sparse_matrix(self.interactions)
        model = TensorRec(n_components=10)
        model.fit(int_as_dataset, self.user_features, self.item_features, epochs=10)

    def test_fit_from_datasets(self):
        uf_as_dataset = create_tensorrec_dataset_from_sparse_matrix(self.user_features)
        if_as_dataset = create_tensorrec_dataset_from_sparse_matrix(self.item_features)
        int_as_dataset = create_tensorrec_dataset_from_sparse_matrix(self.interactions)
        model = TensorRec(n_components=10)
        model.fit(int_as_dataset, uf_as_dataset, if_as_dataset, epochs=10)

    def test_fit_from_tfrecords(self):
        set_session(None)
        model = TensorRec(n_components=10)
        model.fit(self.interactions_path, self.user_features_path, self.item_features_path, epochs=10)


class TensorRecAPITestCase(TestCase):

    @classmethod
    def setUpClass(cls):

        cls.n_users = 15
        cls.n_items = 30

        cls.interactions, cls.user_features, cls.item_features = generate_dummy_data(
            num_users=cls.n_users, num_items=cls.n_items, interaction_density=.5, num_user_features=200,
            num_item_features=200, n_features_per_user=20, n_features_per_item=20, pos_int_ratio=.5
        )

        cls.standard_model = TensorRec(n_components=10)
        cls.standard_model.fit(cls.interactions, cls.user_features, cls.item_features, epochs=10)

        cls.unbiased_model = TensorRec(n_components=10, biased=False)
        cls.unbiased_model.fit(cls.interactions, cls.user_features, cls.item_features, epochs=10)

    def test_fit(self):
        # Ensure that the nodes have been built
        self.assertIsNotNone(self.standard_model.tf_prediction)

    def test_predict(self):
        predictions = self.standard_model.predict(user_features=self.user_features,
                                                  item_features=self.item_features)

        self.assertEqual(predictions.shape, (self.n_users, self.n_items))

    def test_predict_rank(self):
        ranks = self.standard_model.predict_rank(user_features=self.user_features,
                                                 item_features=self.item_features)

        self.assertEqual(ranks.shape, (self.n_users, self.n_items))
        for x in range(ranks.shape[0]):
            for y in range(ranks.shape[1]):
                val = ranks[x][y]
                self.assertGreater(val, 0)

    def test_predict_similar_items(self):
        sims = self.standard_model.predict_similar_items(item_features=self.item_features,
                                                         item_ids=[6, 12],
                                                         n_similar=5)

        # Two items, two rows of sims
        self.assertEqual(len(sims), 2)

        for item_sims in sims:
            # Should equal n_similar
            self.assertEqual(len(item_sims), 5)

    def test_fit_predict_unbiased(self):
        predictions = self.unbiased_model.predict(user_features=self.user_features, item_features=self.item_features)
        self.assertEqual(predictions.shape, (self.n_users, self.n_items))

    def test_predict_user_repr(self):
        user_repr = self.unbiased_model.predict_user_representation(self.user_features)
        self.assertEqual(user_repr.shape, (self.n_users, 10))

    def test_predict_item_repr(self):
        item_repr = self.unbiased_model.predict_item_representation(self.item_features)
        self.assertEqual(item_repr.shape, (self.n_items, 10))

    def test_predict_user_bias_unbiased_model(self):
        self.assertRaises(
            ModelNotBiasedException,
            self.unbiased_model.predict_user_bias,
            self.user_features)

    def test_predict_item_bias_unbiased_model(self):
        self.assertRaises(
            ModelNotBiasedException,
            self.unbiased_model.predict_item_bias,
            self.item_features)

    def test_predict_user_attn_repr(self):
        # This test will be overwritten by the tests that have attention
        with self.assertRaises(ModelWithoutAttentionException):
            self.unbiased_model.predict_user_attention_representation(self.user_features)


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


class TensorRecAPINTastesTestCase(TensorRecAPITestCase):

    @classmethod
    def setUpClass(cls):

        cls.n_users = 15
        cls.n_items = 30

        cls.interactions, cls.user_features, cls.item_features = generate_dummy_data(
            num_users=cls.n_users, num_items=cls.n_items, interaction_density=.5, num_user_features=200,
            num_item_features=200, n_features_per_user=20, n_features_per_item=20, pos_int_ratio=.5
        )

        cls.standard_model = TensorRec(n_components=10,
                                       n_tastes=3,
                                       user_repr_graph=NormalizedLinearRepresentationGraph())
        cls.standard_model.fit(cls.interactions, cls.user_features, cls.item_features, epochs=10)

        cls.unbiased_model = TensorRec(n_components=10,
                                       n_tastes=3,
                                       biased=False,
                                       user_repr_graph=NormalizedLinearRepresentationGraph())
        cls.unbiased_model.fit(cls.interactions, cls.user_features, cls.item_features, epochs=10)

    def test_predict_user_repr(self):
        user_repr = self.unbiased_model.predict_user_representation(self.user_features)

        # 3 tastes, shape[0] users, 10 components
        self.assertEqual(user_repr.shape, (3, self.user_features.shape[0], 10))


class TensorRecAPIAttentionTestCase(TensorRecAPINTastesTestCase):

    @classmethod
    def setUpClass(cls):

        cls.n_users = 15
        cls.n_items = 30

        cls.interactions, cls.user_features, cls.item_features = generate_dummy_data(
            num_users=cls.n_users, num_items=cls.n_items, interaction_density=.5, num_user_features=200,
            num_item_features=200, n_features_per_user=20, n_features_per_item=20, pos_int_ratio=.5
        )

        cls.standard_model = TensorRec(n_components=10,
                                       n_tastes=3,
                                       user_repr_graph=NormalizedLinearRepresentationGraph(),
                                       attention_graph=LinearRepresentationGraph())
        cls.standard_model.fit(cls.interactions, cls.user_features, cls.item_features, epochs=10)

        cls.unbiased_model = TensorRec(n_components=10,
                                       n_tastes=3,
                                       biased=False,
                                       user_repr_graph=NormalizedLinearRepresentationGraph(),
                                       attention_graph=LinearRepresentationGraph())
        cls.unbiased_model.fit(cls.interactions, cls.user_features, cls.item_features, epochs=10)

    def test_predict_user_attn_repr(self):
        user_attn_repr = self.unbiased_model.predict_user_attention_representation(self.user_features)

        # attn repr should have shape [n_tastes, n_users, n_components]
        self.assertEqual(user_attn_repr.shape, (3, self.user_features.shape[0], 10))


class TensorRecAPIDatasetInputTestCase(TensorRecAPITestCase):

    @classmethod
    def setUpClass(cls):

        cls.n_users = 15
        cls.n_items = 30

        cls.interactions, cls.user_features, cls.item_features = generate_dummy_data(
            num_users=cls.n_users, num_items=cls.n_items, interaction_density=.5, num_user_features=200,
            num_item_features=200, n_features_per_user=20, n_features_per_item=20, pos_int_ratio=.5,
            return_datasets=True
        )

        cls.standard_model = TensorRec(n_components=10)
        cls.standard_model.fit(cls.interactions, cls.user_features, cls.item_features, epochs=10)

        cls.unbiased_model = TensorRec(n_components=10, biased=False)
        cls.unbiased_model.fit(cls.interactions, cls.user_features, cls.item_features, epochs=10)


class TensorRecAPITFRecordInputTestCase(TensorRecAPITestCase):

    @classmethod
    def setUpClass(cls):

        # Blow away an existing session to avoid 'tf_map_func not found' error
        set_session(None)

        cls.n_users = 15
        cls.n_items = 30

        int_ds, uf_ds, if_ds = generate_dummy_data(
            num_users=cls.n_users, num_items=cls.n_items, interaction_density=.5, num_user_features=200,
            num_item_features=200, n_features_per_user=20, n_features_per_item=20, pos_int_ratio=.5
        )

        cls.temp_dir = tempfile.mkdtemp()
        cls.interactions = os.path.join(cls.temp_dir, 'interactions.tfrecord')
        cls.user_features = os.path.join(cls.temp_dir, 'user_features.tfrecord')
        cls.item_features = os.path.join(cls.temp_dir, 'item_features.tfrecord')

        write_tfrecord_from_sparse_matrix(cls.interactions, int_ds)
        write_tfrecord_from_sparse_matrix(cls.user_features, uf_ds)
        write_tfrecord_from_sparse_matrix(cls.item_features, if_ds)

        cls.standard_model = TensorRec(n_components=10)
        cls.standard_model.fit(cls.interactions, cls.user_features, cls.item_features, epochs=10)

        cls.unbiased_model = TensorRec(n_components=10, biased=False)
        cls.unbiased_model.fit(cls.interactions, cls.user_features, cls.item_features, epochs=10)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)


class TensorRecSavingTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.interactions, cls.user_features, cls.item_features = generate_dummy_data(
            num_users=15, num_items=30, interaction_density=.5, num_user_features=200, num_item_features=200,
            n_features_per_user=20, n_features_per_item=20, pos_int_ratio=.5
        )
        tf.reset_default_graph()
        set_session(None)

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
        set_session(None)
        tf.reset_default_graph()

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
