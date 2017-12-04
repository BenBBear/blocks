import mxnet as mx
from utils.args import get, check


def get_lr_scheduler(**kwargs):
    begin_epoch = get(kwargs, 'begin_epoch', int, 0)
    epoch_size = get(kwargs, 'epoch_size', int, 1000)
    steps = get(kwargs, 'steps', lambda x: [int(i) for i in x.split('-')], [30, 50])
    decay = get(kwargs, 'decay', float, 0.1)
    check(kwargs, 'lr_scheduler.step', ['begin_epoch', 'epoch_size', 'steps', 'decay'])
    step_ = [epoch_size * (x - begin_epoch)
             for x in steps if x - begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=decay) if len(step_) else None