import mxnet as mx


def get_metric(**kwargs):
    return ['acc', mx.metric.create('top_k_accuracy', top_k = 5)]