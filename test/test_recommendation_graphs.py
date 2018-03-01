import numpy as np
import tensorflow as tf
from unittest import TestCase

from tensorrec.recommendation_graphs import gather_sampled_item_predictions
from tensorrec.session_management import get_session


class RecommendationGraphsTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.session = get_session()

    def test_gather_sampled_item_predictions(self):
        input_data = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12]
        ])
        sample_indices = np.array([
            [0, 3],  # Corresponds to [1, 4]
            [5, 6],  # Corresponds to [6, 7]
            [8, 8],  # Corresponds to [9, 9]
        ])
        result = gather_sampled_item_predictions(
            tf_prediction=tf.identity(input_data), tf_sampled_item_indices=tf.identity(sample_indices)
        ).eval(session=self.session)

        expected_result = np.array([
            [1, 4],
            [6, 7],
            [9, 9],
        ])
        self.assertTrue((result == expected_result).all())
