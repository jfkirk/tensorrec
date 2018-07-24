import math
import numpy as np
from unittest import TestCase

from tensorrec import TensorRec
from tensorrec.prediction_graphs import (
    DotProductPredictionGraph, CosineSimilarityPredictionGraph, EuclidianSimilarityPredictionGraph
)
from tensorrec.session_management import get_session
from tensorrec.util import generate_dummy_data_with_indicator


class PredictionGraphsTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.interactions, cls.user_features, cls.item_features = generate_dummy_data_with_indicator(
            num_users=10, num_items=12, interaction_density=.5)

    def test_dot_product(self):
        model = TensorRec(prediction_graph=DotProductPredictionGraph())
        model.fit(self.interactions, self.user_features, self.item_features, epochs=5)

    def test_cos_distance(self):
        model = TensorRec(prediction_graph=CosineSimilarityPredictionGraph())
        model.fit(self.interactions, self.user_features, self.item_features, epochs=5)


class DotProductTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.session = get_session()

    def test_dense_prediction(self):
        graph = DotProductPredictionGraph()
        array_1 = np.array([
            [1.0, 1.0],
            [10.0, 10.0],
            [-1.0, 1.0],
        ])
        array_2 = np.array([
            [1.0, 1.0],
            [-1.0, -1.0],
            [-1.0, 1.0],
        ])
        result = graph.connect_dense_prediction_graph(tf_user_representation=array_1,
                                                      tf_item_representation=array_2).eval(session=self.session)

        expected_result = np.array([
            [2.0, -2.0, 0.0],
            [20.0, -20.0, 0.0],
            [0.0, 0.0, 2.0]
        ])
        self.assertTrue(np.allclose(result, expected_result))

    def test_serial_prediction(self):
        graph = DotProductPredictionGraph()
        array_1 = np.array([
            [1.0, 1.0],
            [10.0, 10.0],
            [-1.0, 1.0],
        ])
        array_2 = np.array([
            [1.0, 1.0],
            [-1.0, -1.0],
            [-1.0, 1.0],
        ])

        x_user = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        x_item = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])

        result = graph.connect_serial_prediction_graph(tf_user_representation=array_1,
                                                       tf_item_representation=array_2,
                                                       tf_x_user=x_user,
                                                       tf_x_item=x_item,).eval(session=self.session)

        expected_result = np.array([2.0, -2.0, 0.0,
                                    20.0, -20.0, 0.0,
                                    0.0, 0.0, 2.0])
        self.assertTrue(np.allclose(result, expected_result))


class CosineSimilarityTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.session = get_session()

    def test_dense_prediction(self):
        graph = CosineSimilarityPredictionGraph()
        array_1 = np.array([
            [1.0, 1.0],
            [10.0, 10.0],
            [-1.0, 1.0],
        ])
        array_2 = np.array([
            [1.0, 1.0],
            [-1.0, -1.0],
            [-1.0, 1.0],
        ])
        result = graph.connect_dense_prediction_graph(tf_user_representation=array_1,
                                                      tf_item_representation=array_2).eval(session=self.session)

        expected_result = np.array([
            [1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        self.assertTrue(np.allclose(result, expected_result))

    def test_serial_prediction(self):
        graph = CosineSimilarityPredictionGraph()
        array_1 = np.array([
            [1.0, 1.0],
            [10.0, 10.0],
            [-1.0, 1.0],
        ])
        array_2 = np.array([
            [1.0, 1.0],
            [-1.0, -1.0],
            [-1.0, 1.0],
        ])

        x_user = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        x_item = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])

        result = graph.connect_serial_prediction_graph(tf_user_representation=array_1,
                                                       tf_item_representation=array_2,
                                                       tf_x_user=x_user,
                                                       tf_x_item=x_item,).eval(session=self.session)

        expected_result = np.array([1.0, -1.0, 0.0,
                                    1.0, -1.0, 0.0,
                                    0.0, 0.0, 1.0])
        self.assertTrue(np.allclose(result, expected_result))


class EuclidianSimilarityTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.session = get_session()

    def test_dense_prediction(self):
        graph = EuclidianSimilarityPredictionGraph()
        array_1 = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
            [-1.0, 1.0],
        ])
        array_2 = np.array([
            [1.0, 1.0],
            [-1.0, -1.0],
            [-1.0, 1.0],
        ])

        result = graph.connect_dense_prediction_graph(tf_user_representation=array_1,
                                                      tf_item_representation=array_2).eval(session=self.session)

        expected_result = np.array([
            [0.0, -math.sqrt(8.0), -2.0],
            [-math.sqrt(2.0), -math.sqrt(18.0), -math.sqrt(10.0)],
            [-2.0, -2.0, 0.0]]
        )
        self.assertTrue(np.allclose(result, expected_result))

    def test_serial_prediction(self):
        graph = EuclidianSimilarityPredictionGraph()
        array_1 = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
            [-1.0, 1.0],
        ])
        array_2 = np.array([
            [1.0, 1.0],
            [-1.0, -1.0],
            [-1.0, 1.0],
        ])

        x_user = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        x_item = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])

        result = graph.connect_serial_prediction_graph(tf_user_representation=array_1,
                                                       tf_item_representation=array_2,
                                                       tf_x_user=x_user,
                                                       tf_x_item=x_item,).eval(session=self.session)

        expected_result = np.array([0.0, -math.sqrt(8.0), -2.0,
                                    -math.sqrt(2.0), -math.sqrt(18.0), -math.sqrt(10.0),
                                    -2.0, -2.0, 0.0])
        self.assertTrue(np.allclose(result, expected_result))
