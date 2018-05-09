from unittest import TestCase
from numpy.random import shuffle
import numpy as np
import scipy.sparse as sps

from tensorrec import TensorRec
from tensorrec.eval import recall_at_k, precision_at_k, f1_score_at_k, ndcg_at_k
from tensorrec.eval import _setup_ndcg, _idcg, _dcg
from tensorrec.util import generate_dummy_data_with_indicator


class EvalTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.interactions, cls.user_features, cls.item_features = generate_dummy_data_with_indicator(
            num_users=10, num_items=12, interaction_density=.5)
        model = TensorRec(n_components=10)
        model.fit(cls.interactions, cls.user_features, cls.item_features, epochs=10)
        cls.model = model
        cls.ranks = model.predict_rank(user_features=cls.user_features, item_features=cls.item_features)

    def test_recall_at_k(self):
        # Check non-none results with and without preserve_rows
        self.assertIsNotNone(recall_at_k(predicted_ranks=self.ranks,
                                         test_interactions=self.interactions,
                                         k=5,
                                         preserve_rows=False))
        self.assertIsNotNone(recall_at_k(predicted_ranks=self.ranks,
                                         test_interactions=self.interactions,
                                         k=5,
                                         preserve_rows=True))

    def test_precision_at_k(self):
        # Check non-none results with and without preserve_rows
        self.assertIsNotNone(precision_at_k(predicted_ranks=self.ranks,
                                            test_interactions=self.interactions,
                                            k=5,
                                            preserve_rows=False))
        self.assertIsNotNone(precision_at_k(predicted_ranks=self.ranks,
                                            test_interactions=self.interactions,
                                            k=5,
                                            preserve_rows=True))

    def test_f1_score_at_k(self):
        # Check non-none results with and without preserve_rows
        self.assertIsNotNone(f1_score_at_k(predicted_ranks=self.ranks,
                                           test_interactions=self.interactions,
                                           k=5,
                                           preserve_rows=False))
        self.assertIsNotNone(f1_score_at_k(predicted_ranks=self.ranks,
                                           test_interactions=self.interactions,
                                           k=5,
                                           preserve_rows=True))

    def test_ndcg_at_k(self):

        ndcg10 = np.mean(ndcg_at_k(
            predicted_ranks=self.ranks,
            test_interactions=self.interactions,
            k=10,
            preserve_rows=False
        ))

        ndcg20 = np.mean(ndcg_at_k(
            predicted_ranks=self.ranks,
            test_interactions=self.interactions,
            k=20,
            preserve_rows=False
        ))

        ndcg5 = np.mean(ndcg_at_k(
            predicted_ranks=self.ranks,
            test_interactions=self.interactions,
            k=5,
            preserve_rows=False
        ))

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
        self.assertAlmostEqual(exponential_calc, 18.77105, 3)

        ordered_calc = _idcg(ordered)
        self.assertEqual(ordered_calc, exponential_calc)

        binary = np.array([1, 1, 1, 1, 0, 0])
        # By hand
        terms = [(2**e-1)/(np.log2(i+2)) for i, e in enumerate(list(binary))]
        est_idcg = np.sum(terms)
        hits = binary
        self.assertEqual(_idcg(hits), est_idcg)

    def test_ndcg_setup(self):
        wiki_rel = sps.lil_matrix(np.array([3, 2, 3, 0, 1, 2]))
        wiki_rank = np.array([1, 2, 3, 4, 5, 6])

        # at 10
        rel, k_mask, ror, ror_at_k = _setup_ndcg(wiki_rank, wiki_rel)

        self.assertIsNotNone(rel)
        self.assertEqual(len(k_mask), 5)  # all non-zero elements in relevance where rank <= k
        self.assertEqual(len(k_mask), len(ror_at_k))
        ror_cnt = sum([v[0] == v[1] for v in zip(ror.data, [1, 2, 3, 5, 6])])
        self.assertEqual(ror_cnt, len(k_mask))  # check length is k mask and correct index

    def test_dcg(self):
        wiki_rel = sps.lil_matrix(np.array([3, 3, 1, 0, 2]))
        wiki_rank = np.array([1, 2, 3, 4, 5])
        rel, k_mask, ror, ror_at_k = _setup_ndcg(wiki_rank, wiki_rel)

        # by hand...
        mat = np.array([3, 3, 1, 2])
        ranks = np.array([1, 2, 3, 5])

        numer_bh = 2**np.multiply(mat, [1, 1, 1, 1]) - 1
        denom_bh = np.log2(ranks + 1)

        # in function calc
        numer_func = (2 ** np.multiply(rel.data, k_mask)) - 1
        denom_func = np.log2(ror_at_k + 1)

        numer_all_eq = sum([v[0] == v[1] for v in zip(numer_bh, numer_func)])
        self.assertEqual(numer_all_eq, len(mat))

        denom_all_eq = sum([v[0] == v[1] for v in zip(denom_bh, denom_func)])
        self.assertEqual(denom_all_eq, len(mat))

        by_hand_dcg = np.sum(numer_bh/denom_bh)
        func_dcg = _dcg(rel, k_mask, ror_at_k, ror)
        self.assertEqual(by_hand_dcg, func_dcg)

        # functional NDCG should equal example another example
        # http://people.cs.georgetown.edu/~nazli/classes/ir-Slides/Evaluation-13.pdf
        # about .96 but since we use Burges 2005 formulation
        # it will be a little different
        v = func_dcg/_idcg(np.array([3, 3, 1, 0, 2]))
        self.assertAlmostEqual(v.item(0), .979762, 3)
