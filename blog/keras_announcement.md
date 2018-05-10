
# Deep Learning for Recommendation with Keras and TensorRec

Deep Learning for Recommendation with Keras and TensorRec

With the release of [TensorRec](https://github.com/jfkirk/tensorrec) v0.21, I’ve added the ability to easily use deep neural networks in your recommender system.

For some recommender problems, such as cold-start recommendation problems, deep learning can be an elegant solution for learning from user and item metadata. Using [TensorRec](https://github.com/jfkirk/tensorrec) with [Keras](https://keras.io/), you can now experiment with deep representation models in your recommender systems quickly and easily.

## Implementation

In a TensorRec model, the components that learn how to process user and item features are called the “representation graphs” (or “repr” for short). These graphs convert high-dimensional user/item features, such as metadata and indicator variables, into low-dimensional user/item representations.

![The TensorRec recommender system.](https://cdn-images-1.medium.com/max/4260/1*YotDpHjvGL8xK91ZggthbA.png)*The TensorRec recommender system.*

Define your model’s representation graphs as a sequence of Keras layers by extending the class AbstractKerasRepresentationGraph. Next, overwrite the create_layers abstract method to return an ordered list of Keras layers.

<iframe src="https://medium.com/media/508acf0b226c590995540243f6458247" frameborder=0></iframe>

## **Example**

You can find the example above used in a full recommender system for the [Book Crossing dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/) in [this module.](https://github.com/jfkirk/tensorrec/blob/master/examples/keras_example.py)

With this dataset, the recommender system is learning to recommend books to users based on user metadata (location, age) and book metadata (author, title, year publisher, etc). This example compares four different configurations of a recommender system using combinations of linear kernel representations and deep network representations.

![Book Crossing example results](https://cdn-images-1.medium.com/max/2468/1*hSSAFi9z71JOQmM__UjnHA.png)*Book Crossing example results*

The architecture of this network has not been optimized for the Book Crossing problem, and should be regarded only as an example of the new Keras functionality.

## **A Word of Caution**

When you’re holding a hammer (or deep learning), everything looks like a nail (or a deep learning problem).

Many real-world recommendation problems are best solved through thoughtful analysis, feature engineering, simple models, and effective feedback systems. Deep models are highly flexible, but this makes them particularly susceptible to overfitting through overparameterization. In these cases, performance with previously-unknown users and items can be significantly harmed.

An example of this issue is the third configuration in the Book Crossing example above. This system is the most effective of the four configurations at making recommendations for existing users based only on their age and location but, based on the same user metadata, is the worst performer on new users — a classic example of overfitting. This problem could be avoided with richer metadata, better feature engineering, or simpler models.

The Book Crossing example also illustrates that the best product outcomes may be from having multiple recommender systems. A product that is driven by two recommender systems, one designed and optimized for warm-start users and another for new cold-start users, may be the optimal system design for your product.

If you’d like to contribute to (or even just try out) TensorRec I’d love to hear your feedback either on [GitHub](https://github.com/jfkirk/tensorrec) or in the comment section of this post.

Special thanks to Joe Cauteruccio (Spotify), Mike Sanders (Spotify), and Logan Moore (Northeastern University) for their contributions to TensorRec.

*Note: This is a personal project and, at time of writing, is not associated with Spotify.*
