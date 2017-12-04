import mxnet as mx
from utils.args import get, check


def get_lr_scheduler(**kwargs):
    epoch_size = get(kwargs, 'epoch_size', int, 1000)
    period = get(kwargs, 'period', float, 1.0)
    decay = get(kwargs, 'decay', float, 0.1)
    check(kwargs, 'lr_scheduler.factor', ['epoch_size', 'period', 'decay'])
    return mx.lr_scheduler.FactorScheduler(step=int(period * epoch_size), factor=decay)
