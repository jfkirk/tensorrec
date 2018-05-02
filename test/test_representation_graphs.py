from nose_parameterized import parameterized
from unittest import TestCase

from tensorrec import TensorRec
from tensorrec.representation_graphs import (
    LinearRepresentationGraph, NormalizedLinearRepresentationGraph, FeaturePassThroughRepresentationGraph,
    WeightedFeaturePassThroughRepresentationGraph, ReLURepresentationGraph
)
from tensorrec.util import generate_dummy_data


class RepresentationGraphTestCase(TestCase):

    @parameterized.expand([
        ["linear", LinearRepresentationGraph, LinearRepresentationGraph, 50, 60, 20],
        ["norm_lin", NormalizedLinearRepresentationGraph, NormalizedLinearRepresentationGraph, 50, 60, 20],
        ["fpt_user", FeaturePassThroughRepresentationGraph, NormalizedLinearRepresentationGraph, 50, 60, 50],
        ["fpt_item", NormalizedLinearRepresentationGraph, FeaturePassThroughRepresentationGraph, 50, 60, 60],
        ["fpt_both", FeaturePassThroughRepresentationGraph, FeaturePassThroughRepresentationGraph, 50, 50, 50],
        ["weighted_fpt", WeightedFeaturePassThroughRepresentationGraph, WeightedFeaturePassThroughRepresentationGraph,
         50, 50, 50],
        ["relu", ReLURepresentationGraph, ReLURepresentationGraph, 50, 60, 20],
    ])
    def test_fit(self, name, user_repr, item_repr, n_user_features, n_item_features, n_components):
        interactions, user_features, item_features = generate_dummy_data(
            num_users=15, num_items=30, interaction_density=.5, num_user_features=n_user_features,
            num_item_features=n_item_features, n_features_per_user=20, n_features_per_item=20, pos_int_ratio=.5
        )
        model = TensorRec(n_components=n_components,
                          user_repr_graph=user_repr(),
                          item_repr_graph=item_repr())
        model.fit(interactions, user_features, item_features, epochs=10)

        # Ensure that the nodes have been built
        self.assertIsNotNone(model.tf_prediction)


class IdentityRepresentationGraphTestCase(TestCase):

    def test_fit_fail_on_bad_dims(self):
        interactions, user_features, item_features = generate_dummy_data(
            num_users=15, num_items=30, interaction_density=.5, num_user_features=30,
            num_item_features=20, n_features_per_user=20, n_features_per_item=20, pos_int_ratio=.5
        )

        with self.assertRaises(ValueError):
            model = TensorRec(n_components=25,
                              user_repr_graph=FeaturePassThroughRepresentationGraph(),
                              item_repr_graph=LinearRepresentationGraph())
            model.fit(interactions, user_features, item_features, epochs=10)

        with self.assertRaises(ValueError):
            model = TensorRec(n_components=25,
                              user_repr_graph=LinearRepresentationGraph(),
                              item_repr_graph=FeaturePassThroughRepresentationGraph())
            model.fit(interactions, user_features, item_features, epochs=10)
