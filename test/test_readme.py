from unittest import TestCase


class ReadmeTestCase(TestCase):

    def test_basic_usage(self):
        import numpy as np
        import tensorrec

        # Build the model with default parameters
        model = tensorrec.TensorRec()

        # Generate some dummy data
        interactions, user_features, item_features = tensorrec.util.generate_dummy_data(num_users=100,
                                                                                        num_items=150,
                                                                                        interaction_density=.05)

        # Fit the model for 5 epochs
        model.fit(interactions, user_features, item_features, epochs=5, verbose=True)

        # Predict scores and ranks for all users and all items
        predictions = model.predict(user_features=user_features,
                                    item_features=item_features)
        predicted_ranks = model.predict_rank(user_features=user_features,
                                             item_features=item_features)

        # Calculate and print the recall at 10
        r_at_k = tensorrec.eval.recall_at_k(predicted_ranks, interactions, k=10)
        print(np.mean(r_at_k))

        self.assertIsNotNone(predictions)

    def test_custom_repr_graph(self):
        import tensorflow as tf
        import tensorrec

        # Define a custom representation function graph
        class TanhRepresentationGraph(tensorrec.representation_graphs.AbstractRepresentationGraph):
            def connect_representation_graph(self, tf_features, n_components, n_features, node_name_ending):
                """
                This representation function embeds the user/item features by passing them through a single tanh layer.
                :param tf_features: tf.SparseTensor
                The user/item features as a SparseTensor of dimensions [n_users/items, n_features]
                :param n_components: int
                The dimensionality of the resulting representation.
                :param n_features: int
                The number of features in tf_features
                :param node_name_ending: String
                Either 'user' or 'item'
                :return:
                A tuple of (tf.Tensor, list) where the first value is the resulting representation in n_components
                dimensions and the second value is a list containing all tf.Variables which should be subject to
                regularization.
                """
                tf_tanh_weights = tf.Variable(tf.random_normal([n_features, n_components], stddev=.5),
                                              name='tanh_weights_%s' % node_name_ending)

                tf_repr = tf.nn.tanh(tf.sparse_tensor_dense_matmul(tf_features, tf_tanh_weights))

                # Return repr layer and variables
                return tf_repr, [tf_tanh_weights]

        # Build a model with the custom representation function
        model = tensorrec.TensorRec(user_repr_graph=TanhRepresentationGraph(),
                                    item_repr_graph=TanhRepresentationGraph())

        # Generate some dummy data
        interactions, user_features, item_features = tensorrec.util.generate_dummy_data(num_users=100,
                                                                                        num_items=150,
                                                                                        interaction_density=.05)

        # Fit the model for 5 epochs
        model.fit(interactions, user_features, item_features, epochs=5, verbose=True)

        self.assertIsNotNone(model)

    def test_custom_loss_graph(self):
        import tensorflow as tf
        import tensorrec

        # Define a custom loss graph
        class SimpleLossGraph(tensorrec.loss_graphs.AbstractLossGraph):
            def connect_loss_graph(self, tf_prediction_serial, tf_interactions_serial, **kwargs):
                """
                This loss function returns the absolute simple error between the predictions and the interactions.
                :param tf_prediction_serial: tf.Tensor
                The recommendation scores as a Tensor of shape [n_samples, 1]
                :param tf_interactions_serial: tf.Tensor
                The sample interactions corresponding to tf_prediction_serial as a Tensor of shape [n_samples, 1]
                :param kwargs:
                Other TensorFlow nodes.
                :return:
                A tf.Tensor containing the learning loss.
                """
                return tf.reduce_mean(tf.abs(tf_interactions_serial - tf_prediction_serial))

        # Build a model with the custom loss function
        model = tensorrec.TensorRec(loss_graph=SimpleLossGraph())

        # Generate some dummy data
        interactions, user_features, item_features = tensorrec.util.generate_dummy_data(num_users=100,
                                                                                        num_items=150,
                                                                                        interaction_density=.05)

        # Fit the model for 5 epochs
        model.fit(interactions, user_features, item_features, epochs=5, verbose=True)

        self.assertIsNotNone(model)
