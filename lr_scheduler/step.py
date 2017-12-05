import mxnet as mx
from utils.args import get, check


def get_lr_scheduler(**kwargs):
    """
    return a lr_scheduler instance
    :param begin_epoch: the start epoch number (will work with steps)
    :type begin_epoch: int
    :param epoch_size: number of batch for each epoch (for each gpu as well)
    :type epoch_size: int
    :param steps: str like "1-2-3-4", which means decay at epoch 1,2,3,4
    :type steps: str
    :param decay: lr * decay in each period
    :type decay: float
    :return: mx.lr_scheduler.LRScheduler
    """
    begin_epoch = get(kwargs, 'begin_epoch', int, 0)
    epoch_size = get(kwargs, 'epoch_size', int, 1000)
    steps = get(kwargs, 'steps', lambda x: [int(i) for i in x.split('-')], [30, 50])
    decay = get(kwargs, 'decay', float, 0.1)
    check(kwargs, 'lr_scheduler.step', ['begin_epoch', 'epoch_size', 'steps', 'decay'])
    step_ = [epoch_size * (x - begin_epoch)
             for x in steps if x - begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=decay) if len(step_) else None