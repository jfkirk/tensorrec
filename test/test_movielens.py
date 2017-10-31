from unittest import TestCase

from tensorrec import TensorRec

from test.datasets import get_movielens_100k


class MovieLensTestCase(TestCase):

    def test_movie_lens_fit(self):
        """
        This test checks whether the movielens getter works and that the resulting data is viable for fitting/testing a
        TensorRec model.
        """
        train_interactions, test_interactions, user_features, item_features = get_movielens_100k()

        model = TensorRec()
        model.fit(interactions=train_interactions, user_features=user_features, item_features=item_features)
        predictions = model.predict(user_features=user_features,
                                    item_features=item_features)

        self.assertIsNotNone(predictions)
