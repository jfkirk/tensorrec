from unittest import TestCase

from tensorrec import TensorRec
from tensorrec.loss_graphs import rmse_loss, separation_loss, wmrb_loss
from tensorrec.util import generate_dummy_data_with_indicator


class LossGraphsTestCase(TestCase):

	@classmethod
	def setUpClass(cls):
		cls.interactions, cls.user_features, cls.item_features = generate_dummy_data_with_indicator(
			num_users=10, num_items=12, interaction_density=.5)

	def test_rmse_loss(self):
		model = TensorRec(loss_graph=rmse_loss)
		model.fit(self.interactions, self.user_features, self.item_features, epochs=5)

	def test_rmse_loss_biased(self):
		model = TensorRec(loss_graph=rmse_loss, biased=True)
		model.fit(self.interactions, self.user_features, self.item_features, epochs=5)

	def test_separation_loss(self):
		model = TensorRec(loss_graph=separation_loss)
		model.fit(self.interactions, self.user_features, self.item_features, epochs=5)

	def test_separation_loss_biased(self):
		model = TensorRec(loss_graph=separation_loss, biased=True)
		model.fit(self.interactions, self.user_features, self.item_features, epochs=5)

	def test_wmrb_loss(self):
		model = TensorRec(loss_graph=wmrb_loss)
		model.fit(self.interactions, self.user_features, self.item_features, epochs=5)

	def test_wmrb_loss_biased(self):
		model = TensorRec(loss_graph=wmrb_loss, biased=True)
		model.fit(self.interactions, self.user_features, self.item_features, epochs=5)
