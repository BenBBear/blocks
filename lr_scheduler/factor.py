import mxnet as mx
from utils.args import get, check


def get_lr_scheduler(**kwargs):
    """
    return a lr_scheduler instance
    :param epoch_size: number of batch for each epoch (for each gpu as well)
    :type epoch_size: int
    :param period: decay period
    :type period: int
    :param decay: lr * decay in each period
    :type decay: float
    :return: mx.lr_scheduler.LRScheduler
    """
    epoch_size = get(kwargs, 'epoch_size', int, 1000)
    period = get(kwargs, 'period', float, 1.0)
    decay = get(kwargs, 'decay', float, 0.98)
    check(kwargs, 'lr_scheduler.factor', ['epoch_size', 'period', 'decay'])
    return mx.lr_scheduler.FactorScheduler(step=int(period * epoch_size), factor=decay)
