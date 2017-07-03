# WIP: TensorRec
A TensorFlow recommendation algorithm framework in Python.

# Example
```python
import numpy as np
import tensorflow as tf

import tensorrec

# Build the model with default parameters
model = tensorrec.TensorRec()

# Generate some dummy data
interactions, user_features, item_features = tensorrec.util.generate_dummy_data()

# Start a TensorFlow session and fit the model
session = tf.Session()
model.fit(session, interactions, user_features, item_features, verbose=True)

# Calculate and print the recall at 1000
r_at_k = tensorrec.eval.recall_at_k(model, session, interactions, k=1000, user_features=user_features, item_features=item_features)
print(np.mean(r_at_k))
```
