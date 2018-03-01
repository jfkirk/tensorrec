from unittest import TestCase

from tensorrec.util import calculate_batched_alpha


class UtilTestcase(TestCase):

    def test_calculate_batched_alpha(self):

        # Test no-transformation
        n_batches = 1
        alpha = .01
        self.assertEqual(alpha, calculate_batched_alpha(num_batches=n_batches, alpha=alpha))

        # Test two batches
        n_batches = 2
        alpha = .01
        self.assertAlmostEqual(.53074 * alpha,
                               calculate_batched_alpha(num_batches=n_batches, alpha=alpha),
                               places=5)

        # Test a bad number of batches
        with self.assertRaises(ValueError):
            calculate_batched_alpha(num_batches=0, alpha=alpha)
