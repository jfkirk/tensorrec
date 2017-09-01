from unittest import TestCase

import tensorflow as tf

from tensorrec import TensorRec
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
