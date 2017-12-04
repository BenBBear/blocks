from utils.args import get, check


def get_optimizer(**kwargs):
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
