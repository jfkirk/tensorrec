from unittest import TestCase

from tensorrec import TensorRec
from tensorrec.eval import recall_at_k, precision_at_k, f1_score_at_k
from tensorrec.util import generate_dummy_data


class EvalTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.interactions, cls.user_features, cls.item_features = generate_dummy_data(num_users=10,
                                                                                     num_items=10,
                                                                                     interaction_density=.5)
        model = TensorRec(n_components=10)
        model.fit(cls.interactions, cls.user_features, cls.item_features, epochs=10)
        cls.model = model

    def test_recall_at_k(self):
        # Check non-none results with and without preserve_rows
        self.assertIsNotNone(recall_at_k(model=self.model,
                                         test_interactions=self.interactions,
                                         k=5,
                                         user_features=self.user_features,
                                         item_features=self.item_features,
                                         preserve_rows=False))
        self.assertIsNotNone(recall_at_k(model=self.model,
                                         test_interactions=self.interactions,
                                         k=5,
                                         user_features=self.user_features,
                                         item_features=self.item_features,
                                         preserve_rows=True))

    def test_precision_at_k(self):
        # Check non-none results with and without preserve_rows
        self.assertIsNotNone(precision_at_k(model=self.model,
                                            test_interactions=self.interactions,
                                            k=5,
                                            user_features=self.user_features,
                                            item_features=self.item_features,
                                            preserve_rows=False))
        self.assertIsNotNone(precision_at_k(model=self.model,
                                            test_interactions=self.interactions,
                                            k=5,
                                            user_features=self.user_features,
                                            item_features=self.item_features,
                                            preserve_rows=True))

    def test_f1_score_at_k(self):
        # Check non-none results with and without preserve_rows
        self.assertIsNotNone(f1_score_at_k(model=self.model,
                                           test_interactions=self.interactions,
                                           k=5,
                                           user_features=self.user_features,
                                           item_features=self.item_features,
                                           preserve_rows=False))
        self.assertIsNotNone(f1_score_at_k(model=self.model,
                                           test_interactions=self.interactions,
                                           k=5,
                                           user_features=self.user_features,
                                           item_features=self.item_features,
                                           preserve_rows=True))
