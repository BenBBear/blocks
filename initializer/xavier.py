import mxnet as mx
from utils.args import get, check


def get_initializer(**kwargs):
    """
    return an Xavier initializer
    :param rnd_type: sampling distribution
    :type rnd_type: str
    :param factor_type: xavier fan: "in" or "out"
    :type factor_type: str
    :param magnitude: gaussian variance or uniform maximum
    :type magnitude: float
    :return: mxnet.initializer.Initializer
    """
    result = mx.init.Xavier(rnd_type=get(kwargs, 'rnd_type', str, 'gaussian'),
                            factor_type=get(kwargs, 'factor_type', str, 'in'),
                            magnitude=get(kwargs, 'magnitude', float, 2.37))
    check(kwargs, 'initializer.xavier', ['rnd_type', 'factor_type', 'magnitude'])
    return result