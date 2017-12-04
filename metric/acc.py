import mxnet as mx
from mxnet.metric import Accuracy


def get_metric():
    return [Accuracy(output_names=['softmax_output', ], label_names=['softmax_label', ]), ]