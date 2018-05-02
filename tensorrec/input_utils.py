import numpy as np
from scipy import sparse as sp
import tensorflow as tf

from .session_management import get_session


def create_tensorrec_iterator(name):
    """
    Creates a TensorFlow Iterator that is ready for the standard TensorRec data format.
    :param name: str
    The name for this Iterator.
    :return: tf.data.Iterator
    """
    return tf.data.Iterator.from_structure(
            output_types=(tf.int64, tf.int64, tf.float32, tf.int64, tf.int64),
            output_shapes=([None], [None], [None], [], []),
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

    row_index = np.array([sparse_matrix.row], dtype=np.int64)
    col_index = np.array([sparse_matrix.col], dtype=np.int64)
    values = np.array([sparse_matrix.data], dtype=np.float32)
    n_dim_0 = np.array([sparse_matrix.shape[0]], dtype=np.int64)
    n_dim_1 = np.array([sparse_matrix.shape[1]], dtype=np.int64)

    tensor_slices = (row_index, col_index, values, n_dim_0, n_dim_1)

    return tf.data.Dataset.from_tensor_slices(tensor_slices)


def write_tfrecord_from_sparse_matrix(tfrecord_path, sparse_matrix):
    """
    Writes the contents of a sparse matrix to a TFRecord file.
    :param tfrecord_path: str
    :param sparse_matrix: scipy.sparse matrix
    :return: str
    The tfrecord path
    """
    dataset = create_tensorrec_dataset_from_sparse_matrix(sparse_matrix=sparse_matrix)
    return write_tfrecord_from_tensorrec_dataset(tfrecord_path=tfrecord_path,
                                                 dataset=dataset)


def get_dimensions_from_tensorrec_dataset(dataset):
    """
    Given a TensorFlow Dataset in the standard TensorRec format, returns the dimensions of the SparseTensor to be
    populated by the Dataset.
    :param dataset: tf.data.Dataset
    :return: (int, int)
    """
    session = get_session()
    iterator = create_tensorrec_iterator('dims_iterator')
    initializer = iterator.make_initializer(dataset)
    _, _, _, tf_d0, tf_d1 = iterator.get_next()
    session.run(initializer)
    d0, d1 = session.run([tf_d0, tf_d1])
    return d0, d1


def write_tfrecord_from_tensorrec_dataset(tfrecord_path, dataset):
    """
    Writes the contents of a TensorRec Dataset to a TFRecord file.
    :param tfrecord_path: str
    :param dataset: tf.data.Dataset
    :return: str
    The tfrecord path
    """
    session = get_session()
    iterator = create_tensorrec_iterator('dataset_writing_iterator')
    initializer = iterator.make_initializer(dataset)
    tf_row_index, tf_col_index, tf_values, tf_d0, tf_d1 = iterator.get_next()
    session.run(initializer)
    row_index, col_index, values, d0, d1 = session.run([tf_row_index, tf_col_index, tf_values, tf_d0, tf_d1])

    def _int64_feature(int_values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=int_values))

    def _float_feature(float_values):
        return tf.train.Feature(float_list=tf.train.FloatList(value=float_values))

    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    feature = {
        'row_index': _int64_feature(row_index),
        'col_index': _int64_feature(col_index),
        'values': _float_feature(values),
        'd0': _int64_feature([d0]),
        'd1': _int64_feature([d1]),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
    writer.close()
    return tfrecord_path


def create_tensorrec_dataset_from_tfrecord(tfrecord_path):
    """
    Loads a TFRecord file and creates a Dataset with the contents.
    :param tfrecord_path: str
    :return: tf.data.Dataset
    """

    def parse_tensorrec_tfrecord(example_proto):
        features = {
            'row_index': tf.FixedLenSequenceFeature((), tf.int64, allow_missing=True),
            'col_index': tf.FixedLenSequenceFeature((), tf.int64, allow_missing=True),
            'values': tf.FixedLenSequenceFeature((), tf.float32, allow_missing=True),
            'd0': tf.FixedLenFeature((), tf.int64),
            'd1': tf.FixedLenFeature((), tf.int64),
        }
        parsed_features = tf.parse_single_example(example_proto, features)
        return (parsed_features['row_index'], parsed_features['col_index'], parsed_features['values'],
                parsed_features['d0'], parsed_features['d1'])

    dataset = tf.data.TFRecordDataset(tfrecord_path).map(parse_tensorrec_tfrecord)
    return dataset
