import mxnet as mx
from mxnet.metric import Accuracy


def get_metric(**kwargs):
    """
    return a classification(top-1) metric
    :return: when it return a dict, it should looks like {'eval_metric', 'validation_metric'},
             corresponds to the parameter in module.fit (different metric used in train / val)
    :type: Union[mx.metric.EvalMetric, list of mx.metric.EvalMetric, dict of str and mx.metric.EvalMetric]
    """
    return [Accuracy(output_names=['softmax_output', ], label_names=['softmax_label', ]), ]