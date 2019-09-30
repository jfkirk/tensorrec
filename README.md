# TensorRec
A TensorFlow recommendation algorithm and framework in Python.

[![PyPI version](https://badge.fury.io/py/tensorrec.svg)](https://badge.fury.io/py/tensorrec) [![Build Status](https://travis-ci.org/jfkirk/tensorrec.svg?branch=master)](https://travis-ci.org/jfkirk/tensorrec) [![Gitter chat](https://badges.gitter.im/tensorrec/gitter.png)](https://gitter.im/tensorrec)

## NOTE: TensorRec is not under active development

TensorRec will not be receiving any more planned updates. Please feel free to open pull requests -- I am happy to review them.

Thank you for your contributions, support, and usage of TensorRec!

-James Kirk, @jfkirk

For similar tools, check out:

[TensorFlow Ranking](https://github.com/tensorflow/ranking/)

[Spotlight](https://github.com/maciejkula/spotlight)

[LightFM](https://github.com/lyst/lightfm)

## What is TensorRec?
TensorRec is a Python recommendation system that allows you to quickly develop recommendation algorithms and customize them using TensorFlow.

TensorRec lets you to customize your recommendation system's representation/embedding functions and loss functions while TensorRec handles the data manipulation, scoring, and ranking to generate recommendations.

A TensorRec system consumes three pieces of data: `user_features`, `item_features`, and `interactions`. It uses this data to learn to make and rank recommendations.

For an overview of TensorRec and its usage, please see the [wiki.](https://github.com/jfkirk/tensorrec/wiki)

For more information, and for an outline of this project, please read [this blog post.](https://medium.com/@jameskirk1/tensorrec-a-recommendation-engine-framework-in-tensorflow-d85e4f0874e8)

For an introduction to building recommender systems, please see [these slides.](https://www.slideshare.net/JamesKirk58/boston-ml-architecting-recommender-systems)

![TensorRec System Diagram](https://raw.githubusercontent.com/jfkirk/tensorrec/master/examples/system_diagram.png)

### Example: Basic usage
```python
import numpy as np
import tensorrec

# Build the model with default parameters
model = tensorrec.TensorRec()

# Generate some dummy data
interactions, user_features, item_features = tensorrec.util.generate_dummy_data(
    num_users=100,
    num_items=150,
    interaction_density=.05
)

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
```

## Quick Start
TensorRec can be installed via pip:
```pip install tensorrec```
