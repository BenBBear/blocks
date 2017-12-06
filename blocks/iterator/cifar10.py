import mxnet as mx
from os.path import join
from constant import CIFAR10
from utils.args import get, check
from utils.parse_args import str2bool


def get_iterator(**kwargs):
    """
    return a tuple(pair) of iterator for cifar10
    :param batch_size: e.g. 100
    :type batch_size: int
    :param kvstore: kvstore instance
    :type: mx.kvstore.KVStore
    :param shuffle: whether to shuffle the training set
    :type shuffle: bool
    :param crop: whether to random crop the training image
    :type crop: bool
    :param mirror: whether to random mirror(flip) the training image
    :type mirror: bool
    :return: (train_iter, val_iter)
    :type: (mxnet.io.DataIter, mxnet.io.DataIter)
    """
    scope_name = "iterator.cifar10"
    batch_size = get(kwargs, 'batch_size', int, 100)
    kvstore = get(kwargs, 'kvstore', lambda x: x, None)
    assert kvstore is not None, 'kvstore is None in ' + scope_name

    shuffle = get(kwargs, 'shuffle', str2bool, True)
    crop = get(kwargs, 'crop', str2bool, True)
    mirror = get(kwargs, 'mirror', str2bool, True)
    check(kwargs, scope_name, ['batch_size', 'shuffle', 'crop', 'mirror', 'kvstore'])

    kargs = dict(
        data_shape=(3, 28, 28),
        # Use mean and scale works equally well with a BatchNorm after data layer
        # here we use this simple method
        mean_r=128,
        mean_g=128,
        mean_b=128,
        scale=0.008,
    )
    train = mx.io.ImageRecordIter(
        path_imgrec=CIFAR10['train'],
        batch_size=batch_size,
        rand_crop=crop,
        rand_mirror=mirror,
        num_parts=kvstore.num_workers,
        part_index=kvstore.rank,
        shuffle_chunk_size=10,
        shuffle=shuffle,
        **kargs
    )
    val = mx.io.ImageRecordIter(
        path_imgrec=CIFAR10['val'],
        rand_crop=False,
        rand_mirror=False,
        batch_size=batch_size,
        num_parts=kvstore.num_workers,
        part_index=kvstore.rank,
        shuffle=False,
        **kargs
    )
    return train, val


get_iterator.num_train_examples = 50000