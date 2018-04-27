import numpy as np
from scipy import sparse as sp
import six
import tensorflow as tf


def create_tensorrec_iterator(name):
    """
    Creates a TensorFlow Iterator that is ready for the standard TensorRec data format.
    :param name: str
    The name for this Iterator.
    :return: tf.data.Iterator
    """
    return tf.data.Iterator.from_structure(
            output_types=(tf.int64, tf.float32, tf.int64, tf.int64),
            output_shapes=([None, 2], [None], [], []),
            shared_name=name
    )


def create_tensorrec_dataset_from_sparse_matrix(sparse_matrix):
    """
    Creates a TensorFlow Dataset containing the data from the given sparse matrix.
    :param sparse_matrix: scipy.sparse matrix
    The data to be contained in this Dataset.
    :return: tf.data.Dataset
    """
    if not isinstance(sparse_matrix, sp.coo_matrix):
        sparse_matrix = sp.coo_matrix(sparse_matrix)

    indices = np.array([[pair for pair in six.moves.zip(sparse_matrix.row, sparse_matrix.col)]], dtype=np.int64)
    values = np.array([sparse_matrix.data], dtype=np.float32)
    n_dim_0 = np.array([sparse_matrix.shape[0]], dtype=np.int64)
    n_dim_1 = np.array([sparse_matrix.shape[1]], dtype=np.int64)

    tensor_slices = (indices, values, n_dim_0, n_dim_1)

    return tf.data.Dataset.from_tensor_slices(tensor_slices)


def get_dimensions_from_tensorrec_dataset(dataset, session):
    """
    Given a TensorFlow Dataset in the standard TensorRec format, returns the dimensions of the SparseTensor to be
    populated by the Dataset.
    :param dataset: tf.data.Dataset
    :param session: tf.Session
    :return: (int, int)
    """
    iterator = create_tensorrec_iterator('dims_iterator')
    initializer = iterator.make_initializer(dataset)
    _, _, tf_d0, tf_d1 = iterator.get_next()

    session.run(initializer)
    d0, d1 = session.run([tf_d0, tf_d1])

    return d0, d1
