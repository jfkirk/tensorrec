from unittest import TestCase

from tensorrec import TensorRec
from tensorrec.loss_graphs import (
    RMSELossGraph, RMSEDenseLossGraph, SeparationLossGraph, WMRBLossGraph, WMRBAlignmentLossGraph
)
from tensorrec.util import generate_dummy_data_with_indicator


class LossGraphsTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.interactions, cls.user_features, cls.item_features = generate_dummy_data_with_indicator(
            num_users=10, num_items=12, interaction_density=.5)

    def test_rmse_loss(self):
        model = TensorRec(loss_graph=RMSELossGraph)
        model.fit(self.interactions, self.user_features, self.item_features, epochs=5)

    def test_rmse_loss_biased(self):
        model = TensorRec(loss_graph=RMSELossGraph, biased=True)
        model.fit(self.interactions, self.user_features, self.item_features, epochs=5)

    def test_rmse_loss(self):
        model = TensorRec(loss_graph=RMSEDenseLossGraph)
        model.fit(self.interactions, self.user_features, self.item_features, epochs=5)

    def test_rmse_loss_biased(self):
        model = TensorRec(loss_graph=RMSEDenseLossGraph, biased=True)
        model.fit(self.interactions, self.user_features, self.item_features, epochs=5)

    def test_separation_loss(self):
        model = TensorRec(loss_graph=SeparationLossGraph)
        model.fit(self.interactions, self.user_features, self.item_features, epochs=5)

    def test_separation_loss_biased(self):
        model = TensorRec(loss_graph=SeparationLossGraph, biased=True)
        model.fit(self.interactions, self.user_features, self.item_features, epochs=5)

    def test_wmrb_loss(self):
        model = TensorRec(loss_graph=WMRBLossGraph)
        model.fit(self.interactions, self.user_features, self.item_features, epochs=5)

    def test_wmrb_loss_biased(self):
        model = TensorRec(loss_graph=WMRBLossGraph, biased=True)
        model.fit(self.interactions, self.user_features, self.item_features, epochs=5)

    def test_wmrb_alignment_loss(self):
        model = TensorRec(loss_graph=WMRBAlignmentLossGraph)
        model.fit(self.interactions, self.user_features, self.item_features, epochs=5)

    def test_wmrb_alignment_loss_biased(self):
        model = TensorRec(loss_graph=WMRBAlignmentLossGraph, biased=True)
        model.fit(self.interactions, self.user_features, self.item_features, epochs=5)
