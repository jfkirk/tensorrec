# WIP: TensorRec
A TensorFlow recommendation algorithm framework in Python.

## Example
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

# Predict scores for user 75 on items 1000, 1001, and 1002
predictions = model.predict(session, 
                            user_ids=[75], 
                            item_ids=[1000, 1001, 1002], 
                            user_features=user_features, 
                            item_features=item_features)

# Calculate and print the recall at 1000
r_at_k = tensorrec.eval.recall_at_k(model, session, interactions, 
                                    k=1000, 
                                    user_features=user_features, 
                                    item_features=item_features)
print(np.mean(r_at_k))
```

## Example: Defining custom representation graph
```python
import tensorflow as tf
import tensorrec

# Define a representation graph function
def build_tanh_representation_graph(tf_features, no_components, n_features, node_name_ending):
    tf_tanh_weights = tf.Variable(tf.random_normal([n_features, no_components], 
                                                   stddev=.5),
                                  name='tanh_weights_%s' % node_name_ending)

    tf_repr = tf.nn.tanh(tf.sparse_tensor_dense_matmul(tf_features, tf_tanh_weights))

    # Return repr layer and variables
    return tf_repr, [tf_tanh_weights]

# Build the model with the custom repr graph
model = tensorrec.TensorRec(user_repr_graph_factory=build_tanh_representation_graph,
                            item_repr_graph_factory=build_tanh_representation_graph)
```
