import mxnet as mx
from utils.args import get, check


def get_initializer(**kwargs):
    result = mx.init.Xavier(rnd_type=get(kwargs, 'rnd_type', str, 'gaussian'),
                            factor_type=get(kwargs, 'factor_type', str, 'in'),
                            magnitude=get(kwargs, 'magnitude', float, 2.37))
    check(kwargs, 'initializer.xavier', ['rnd_type', 'factor_type', 'magnitude'])
    return result