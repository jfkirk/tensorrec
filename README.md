# TensorRec
A TensorFlow recommendation algorithm and framework in Python.

[![PyPI version](https://badge.fury.io/py/tensorrec.svg)](https://badge.fury.io/py/tensorrec) [![Build Status](https://travis-ci.org/jfkirk/tensorrec.svg?branch=master)](https://travis-ci.org/jfkirk/tensorrec) [![Gitter chat](https://badges.gitter.im/tensorrec/gitter.png)](https://gitter.im/tensorrec)

## What is TensorRec?
TensorRec is a Python recommendation system that allows you to quickly develop recommendation algorithms and customize them using TensorFlow.

TensorRec lets you to customize your recommendation system's embedding functions and loss functions while TensorRec handles the data manipulation, scoring, and ranking to generate recommendations.

A TensorRec system consumes three pieces of data: `user_features`, `item_features`, and `interactions`. It uses this data to learn to make and rank recommendations.

For more information, and for an outline of this project, please read [this blog post](https://medium.com/@jameskirk1/tensorrec-a-recommendation-engine-framework-in-tensorflow-d85e4f0874e8).

## Quick Start
TensorRec can be installed via pip:
```pip install tensorrec```

### Example: Basic usage
```python
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

# Predict scores for all users and all items
predictions = model.predict(user_features=user_features,
                            item_features=item_features)

# Calculate and print the recall at 10
r_at_k = tensorrec.eval.recall_at_k(model, interactions,
                                    k=10,
                                    user_features=user_features,
                                    item_features=item_features)
print(np.mean(r_at_k))
```

## Input data

The following examples show what user/item features and interactions would look like in a TensorRec system meant to recommend business consulting projects (items) to consultants (users).

The data is represented in matrices. TensorRec can consume these matrices as any `scipy.sparse` matrix.

### User features:

![user_features](https://cdn-images-1.medium.com/max/1600/1*43Be-sAmktN9HYvseA3mng.png)

### Item features:

![item_features](https://cdn-images-1.medium.com/max/1600/1*56TwD4Sh5A2SEGvl1S_82g.png)

### Interactions:

![interactions](https://cdn-images-1.medium.com/max/1600/1*tfnTAxGB-SSY8tV_Mrw2CQ.png)

Images from [Medium](https://medium.com/product-at-catalant-technologies/using-lightfm-to-recommend-projects-to-consultants-44084df7321c)

## Prediction Graphs
TensorRec allows you to define the algorithm that will be used to compute recommendation scores from a pair of latent representations of your users and items.  
You can define a custom prediction function yourself, or you can use a pre-made prediction function that comes with TensorRec in [tensorrec.prediction_graphs](tensorrec/prediction_graphs.py). 

#### DotProductPredictionGraph
This prediction function calculates the prediction as the dot product between the user and item representations.  
`Prediction = user_repr * item_repr`

#### CosineSimilarityPredictionGraph
This prediction function calculates the prediction as the cosine between the user and item representations.  
`Prediction = cos(user_repr, item_repr)`

#### EuclidianSimilarityPredictionGraph
This prediction function calculates the prediction as the negative euclidian distance between the user and item representations.  
`Prediction = -1 * sqrt(sum((user_repr - item_repr)^2))`

## Representation Graphs
TensorRec allows you to define the algorithm that will be used to compute latent representations (also known as embeddings) of your users and items. You can define a custom representation function yourself, or you can use a pre-made representation function that comes with TensorRec in [tensorrec.representation_graphs](tensorrec/representation_graphs.py).

#### LinearRepresentationGraph
Calculates the representation by passing the features through a linear embedding.

#### ReLURepresentationGraph
Calculates the repesentations by passing the features through a single-layer ReLU neural network.

#### AbstractKerasRepresentationGraph
This abstract RepresentationGraph allows you to use Keras layers as a representation function by overriding the `create_layers()` method.  
An example of this can be found in `examples/keras_example.py`.

### Example: Defining custom representation function
```python
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
```

## Loss Graphs
TensorRec allows you to define the algorithm that will be used to compute loss for a set of recommendation predictions.  
You can define a custom loss function yourself, or you can use a pre-made loss function that comes with TensorRec in [tensorrec.loss_graphs](tensorrec/loss_graphs.py).

#### RMSELossGraph
This loss function returns the root mean square error between the predictions and the true interactions.  
Interactions can be any positive or negative values, and this loss function is sensitive to magnitude.

#### RMSEDenseLossGraph:
This loss function returns the root mean square error between the predictions and the true interactions, including all non-interacted values as 0s.  
Interactions can be any positive or negative values, and this loss function is sensitive to magnitude.

#### SeparationLossGraph
This loss function models the explicit positive and negative interaction predictions as normal distributions and returns the probability of overlap between the two distributions.  
Interactions can be any positive or negative values, but this loss function ignores the magnitude of the interaction -- interactions are grouped in to `{i <= 0}` and `{i > 0}`.

#### SeparationDenseLossGraph
This loss function models all positive and negative interaction predictions as normal distributions and returns the probability of overlap between the two distributions. This loss function includes non-interacted items as negative interactions.  
Interactions can be any positive or negative values, but this loss function ignores the magnitude of the interaction -- interactions are grouped in to `{i <= 0}` and `{i > 0}`.

#### WMRBLossGraph
Approximation of [WMRB: Learning to Rank in a Scalable Batch Training Approach](http://ceur-ws.org/Vol-1905/recsys2017_poster3.pdf) . 
Interactions can be any positive values, but magnitude is ignored. Negative interactions are also ignored.

### Example: Defining custom loss function
```python
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
```
