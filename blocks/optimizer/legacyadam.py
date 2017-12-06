from utils.args import get, check


def get_optimizer(**kwargs):
    """
    return a legacyadam optimizer, it works like
    https://stackoverflow.com/questions/44452571/what-is-the-proper-way-to-weight-decay-for-adam-optimizer
    :param lr: learning rate
    :param lr: float
    :param wd: weight decay factor
    :type wd: float
    :param clip: gradient clip
    :type clip: Union[None, float]
    :return: mxnet.optimizer.Optimizer
    """
    result = {
        "optimizer": "legacyadam",
        "optimizer_params": {
            'learning_rate': get(kwargs, "lr", float, 0.1),
            'wd': get(kwargs, "wd", float, 5e-4),
            'clip_gradient': get(kwargs, "clip", float, None)
        }
    }
    check(kwargs, 'optimizer.legacyadam', ['lr', 'wd', 'clip'])
    return result
