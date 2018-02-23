import tensorflow as tf


def rmse_loss(tf_prediction_serial, tf_interactions_serial, **kwargs):
	"""
	This loss function returns the root mean square error between the predictions and the true interactions.
	Interactions can be any positive or negative values, and this loss function is sensitive to magnitude.
	:param tf_prediction_serial:
	:param tf_interactions_serial:
	:return:
	"""
	return tf.sqrt(tf.reduce_mean(tf.square(tf_interactions_serial - tf_prediction_serial)))


def separation_loss(tf_prediction_serial, tf_interactions_serial, **kwargs):
	"""
	This loss function models the explicit positive and negative interaction predictions as normal distributions and
	returns the probability of overlap between the two distributions.
	Interactions can be any positive or negative values, but this loss function ignored the magnitude of the
	interaction -- interactions are grouped in to {i < 0} and {i > 0}.
	:param tf_prediction_serial:
	:param tf_interactions_serial:
	:return:
	"""

	tf_positive_mask = tf.greater(tf_interactions_serial, 0.0)
	tf_negative_mask = tf.less_equal(tf_interactions_serial, 0.0)

	tf_positive_predictions = tf.boolean_mask(tf_prediction_serial, tf_positive_mask)
	tf_negative_predictions = tf.boolean_mask(tf_prediction_serial, tf_negative_mask)

	tf_pos_mean, tf_pos_var = tf.nn.moments(tf_positive_predictions, axes=[0])
	tf_neg_mean, tf_neg_var = tf.nn.moments(tf_negative_predictions, axes=[0])

	tf_overlap_distribution = tf.contrib.distributions.Normal(loc=(tf_neg_mean - tf_pos_mean),
															  scale=tf.sqrt(tf_neg_var + tf_pos_var))

	loss = 1.0 - tf_overlap_distribution.cdf(0.0)
	return loss


def wmrb_loss(tf_interactions, tf_prediction, **kwargs):
	"""
	Approximation of http://ceur-ws.org/Vol-1905/recsys2017_poster3.pdf
	Interactions can be any positive values, but magnitude is ignored. Negative interactions are also ignored.
	:param tf_interactions:
	:param tf_prediction:
	:param kwargs:
	:return:
	"""

	positive_interaction_mask = tf.greater(tf_interactions.values, 0.0)
	positive_interaction_indices = tf.boolean_mask(tf_interactions.indices,
												   positive_interaction_mask)
	positive_predictions = tf.gather_nd(tf_prediction, indices=positive_interaction_indices)

	n_items = tf.cast(tf.shape(tf_prediction)[1], dtype=tf.float32)
	predictions_sum_per_user = tf.reduce_sum(tf_prediction, axis=1)
	mapped_predictions_sum_per_user = tf.gather(params=predictions_sum_per_user,
												indices=tf.transpose(positive_interaction_indices)[0])

	# TODO smart irrelevant item indicator -- using n_items is an approximation for sparse interactions
	irrelevant_item_indicator = n_items # noqa

	sampled_margin_rank = (n_items - (n_items * positive_predictions)
						   + mapped_predictions_sum_per_user + irrelevant_item_indicator)

	# JKirk - I am leaving out the log term due to experimental results
	# loss = tf.log(sampled_margin_rank + 1.0)
	return sampled_margin_rank


def warp_loss(tf_prediction, tf_y, **kwargs):
	# TODO JK: implement WARP loss

	tf_positive_mask = tf.greater(tf_y, 0.0)
	tf_negative_mask = tf.less_equal(tf_y, 0.0)

	tf_positive_predictions = tf.boolean_mask(tf_prediction, tf_positive_mask) # noqa
	tf_negative_predictions = tf.boolean_mask(tf_prediction, tf_negative_mask) # noqa
