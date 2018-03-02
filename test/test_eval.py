from unittest import TestCase
from numpy.random import shuffle
import numpy as np

from tensorrec import TensorRec
from tensorrec.eval import recall_at_k, precision_at_k, f1_score_at_k
from tensorrec.eval import normalized_discounted_cumulative_gain
from tensorrec.eval import _idcg
from tensorrec.util import generate_dummy_data_with_indicator


class EvalTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.interactions, cls.user_features, cls.item_features = generate_dummy_data_with_indicator(
            num_users=10, num_items=12, interaction_density=.5)
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

    def test_ndcg_at_k(self):

        ndcg10 = normalized_discounted_cumulative_gain(model=self.model,
                                                       test_interactions=self.interactions,
                                                       k=10,
                                                       user_features=self.user_features,
                                                       item_features=self.item_features,
                                                       preserve_rows=False)

        ndcg20 = normalized_discounted_cumulative_gain(model=self.model,
                                                       test_interactions=self.interactions,
                                                       k=20,
                                                       user_features=self.user_features,
                                                       item_features=self.item_features,
                                                       preserve_rows=False)

        ndcg5 = normalized_discounted_cumulative_gain(model=self.model,
                                                      test_interactions=self.interactions,
                                                      k=5,
                                                      user_features=self.user_features,
                                                      item_features=self.item_features,
                                                      preserve_rows=False)

        self.assertIsNotNone(ndcg10)
        self.assertGreaterEqual(ndcg10, ndcg5)
        self.assertEqual(ndcg20, ndcg10)
        self.assertLess(ndcg20, 1)

    def test_idcg_at_k(self):

        ordered = np.array([3, 3, 3, 2, 2, 2, 1, 0])
        wiki_idcg = np.array([3, 3, 3, 2, 2, 2, 1, 0])

        # note different than on page calculation, they use linear
        shuffle(wiki_idcg)
        exponential_calc = _idcg(wiki_idcg)
        self.assertAlmostEqual(exponential_calc, 9.07359, 3)

        ordered_calc = _idcg(ordered)
        self.assertEqual(ordered_calc, exponential_calc)

        binary = np.array([1, 1, 1, 1, 0, 0])
        # By hand
        terms = [e/(np.log2(i+2)) for i, e in enumerate(list(binary))]
        est_idcg = np.sum(terms)
        hits = binary
        self.assertEqual(_idcg(hits), est_idcg)
