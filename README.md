# TensorRec
A TensorFlow recommendation algorithm and framework in Python.

[![Build Status](https://travis-ci.org/jfkirk/tensorrec.svg?branch=master)](https://travis-ci.org/jfkirk/tensorrec) [![Gitter chat](https://badges.gitter.im/tensorrec/gitter.png)](https://gitter.im/tensorrec)

## What is TensorRec?
TensorRec is a Python recommendation system that allows you to quickly develop recommendation algorithms and customize them using TensorFlow.

TensorRec lets you to customize your recommendation system's embedding functions and loss functions while TensorRec handles the data manipulation, scoring, and ranking to generate recommendations.

A TensorRec system consumes three pieces of data: `user_features`, `item_features`, and `interactions`. It uses this data to learn to make and rank recommendations.

For more information, and for an outline of this project, please read [this blog post](https://medium.com/@jameskirk1/tensorrec-a-recommendation-engine-framework-in-tensorflow-d85e4f0874e8).

## Quick Start
TensorRec can be installed via pip:
```pip install tensorrec```

## TODO
Immediate plans for development of TensorRec include:
1. Documentation of TensorRec class and methods
2. Implementation of WARP loss, or an alternate pairwise loss solution
3. Implementation of more evaluation methods (AUC, F score, etc)
4. Integration of publicly available data sets (MovieLens, etc)

## Example: Input data

The following examples show what user/item features and interactions would look like in a TensorRec system meant to recommend business consulting projects (items) to consultants (users).

The data is represented in matrices. TensorRec can consume these matrices as any `scipy.sparse` matrix.

### User features:

![user_features](https://cdn-images-1.medium.com/max/1600/1*43Be-sAmktN9HYvseA3mng.png)

### Item features:

![item_features](https://cdn-images-1.medium.com/max/1600/1*56TwD4Sh5A2SEGvl1S_82g.png)

### Interactions:

![interactions](https://cdn-images-1.medium.com/max/1600/1*tfnTAxGB-SSY8tV_Mrw2CQ.png)

Images from [Medium](https://medium.com/product-at-catalant-technologies/using-lightfm-to-recommend-projects-to-consultants-44084df7321c)

## Example: Basic usage
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

# Predict scores for user 75 on items 100, 101, and 102
predictions = model.predict(user_ids=[75, 75, 75],
                            item_ids=[100, 101, 102],
                            user_features=user_features,
                            item_features=item_features)

# Calculate and print the recall at 10
r_at_k = tensorrec.eval.recall_at_k(model, interactions,
                                    k=10,
                                    user_features=user_features,
                                    item_features=item_features)
print(np.mean(r_at_k))
```

## Example: Defining custom representation function
```python
import tensorflow as tf
import tensorrec

# Define a custom representation function graph
def tanh_representation_graph(tf_features, n_components, n_features, node_name_ending):
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
    tf_tanh_weights = tf.Variable(tf.random_normal([n_features, n_components],
                                                   stddev=.5),
                                  name='tanh_weights_%s' % node_name_ending)

    tf_repr = tf.nn.tanh(tf.sparse_tensor_dense_matmul(tf_features, tf_tanh_weights))

    # Return repr layer and variables
    return tf_repr, [tf_tanh_weights]

# Build a model with the custom representation function
model = tensorrec.TensorRec(user_repr_graph=tanh_representation_graph,
                            item_repr_graph=tanh_representation_graph)
```

## Example: Defining custom loss function
```python
import tensorflow as tf
import tensorrec

# Define a custom loss function graph
def simple_error_graph(tf_prediction, tf_y, **kwargs):
    """
    This loss function returns the absolute simple error between the predictions and the interactions.
    :param tf_prediction: tf.Tensor
    The recommendation scores as a Tensor of shape [n_samples, 1]
    :param tf_y: tf.Tensor
    The sample interactions corresponding to tf_prediction as a Tensor of shape [n_samples, 1]
    :param kwargs:
    Other TensorFlow nodes (not yet implemented)
    :return:
    A tf.Tensor containing the learning loss.
    """
    return tf.reduce_mean(tf.abs(tf_y - tf_prediction))

# Build a model with the custom loss function
model = tensorrec.TensorRec(loss_graph=simple_error_graph)
```
