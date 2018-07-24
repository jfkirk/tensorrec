from .util import lazyval


class TensorRecException(Exception):
    msg = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @lazyval
    def message(self):
        return str(self)

    def __str__(self):
        msg = self.msg.format(**self.kwargs)
        return msg

    __unicode__ = __str__
    __repr__ = __str__


class ModelNotBiasedException(TensorRecException):
    msg = 'Cannot predict {actor} bias for unbiased model'


class ModelNotFitException(TensorRecException):
    msg = "{method}() has been called before model fitting. Call fit() or fit_partial() before calling {method}()."


class ModelWithoutAttentionException(TensorRecException):
    msg = "This TensorRec model does not use attention. Try re-building TensorRec with a valid 'attention_graph' arg."


class BatchNonSparseInputException(TensorRecException):
    msg = 'In order to support user batching at fit time, interactions and user_features must both be scipy.sparse ' \
          'matrices.'


class TfVersionException(TensorRecException):
    msg = "You need to have at least TensorFlow version 1.7 installed in order to use TensorRec properly. You have " \
          "currently installed TensorFlow: {tf_version}"
