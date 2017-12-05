import mxnet as mx
from os.path import join
from constant import IMAGENET_256
from utils.args import get, check
from utils.parse_args import str2bool


def get_iterator(**kwargs):
    """
    return a tuple(pair) of iterator for imagenet
    :param train: hdfs path for train.rec (default is a rec packed from 256x256)
    :type train: str
    :param val: hdfs path for val.rec (default is a rec packed from 256x256)
    :param val: str
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
    :param aug_level: data augmentation level, 1 or 2, 2 will enable aspect/ratio/shear/lumination randomization
    :type aug_level: int
    :return: (train_iter, val_iter)
    :type: (mxnet.io.DataIter, mxnet.io.DataIter)
    """
    scope_name = "iterator.imagenet"
    batch_size = get(kwargs, 'batch_size', int, 100)
    kvstore = get(kwargs, 'kvstore', lambda x: x, None)
    assert kvstore is not None, 'kvstore is None in ' + scope_name

    shuffle = get(kwargs, 'shuffle', str2bool, True)
    crop = get(kwargs, 'crop', str2bool, True)
    mirror = get(kwargs, 'mirror', str2bool, True)
    size = get(kwargs, 'size', int, 224)
    train = get(kwargs, 'train', str, IMAGENET_256["train"])
    val = get(kwargs, 'val', str, IMAGENET_256["val"])
    aug_level = get(kwargs, "aug_level", int, 1)
    check(kwargs, scope_name, ['batch_size', 'shuffle', 'crop', 'mirror', 'kvstore', 'size', 'aug_level', 'train', 'val'])

    train = mx.io.ImageRecordIter(
        path_imgrec=train,
        mean_r=128,  # 123.68,
        mean_g=128,  # 116.779,
        mean_b=128,  # 103.939,
        scale=0.00787,
        round_batch=True,
        data_shape=(3, size, size),
        batch_size=batch_size,
        fill_value=127,
        rand_crop=crop,
        rand_mirror=mirror,
        shuffle=shuffle,
        preprocess_threads=6,
        prefetch_buffer=20,
        prefetch_buffer_keep=10,

        min_random_scale=1.0 if aug_level == 1 else 0.533,  # 256.0/480.0
        max_aspect_ratio=0 if aug_level == 1 else 0.25,
        random_h=0 if aug_level == 1 else 36,  # 0.4*90
        random_s=0 if aug_level == 1 else 50,  # 0.4*127
        random_l=0 if aug_level == 1 else 50,  # 0.4*127
        max_rotate_angle=0 if aug_level <= 2 else 10,
        max_shear_ratio=0 if aug_level <= 2 else 0.1,

        num_parts=kvstore.num_workers,
        part_index=kvstore.rank)

    val = mx.io.ImageRecordIter(
        path_imgrec=val,
        mean_r=128,  # 123.68,
        mean_g=128,  # 116.779,
        mean_b=128,  # 103.939,
        scale=0.00787,

        round_batch=True,
        batch_size=batch_size,
        data_shape=(3, size, size),
        rand_crop=False,
        rand_mirror=False,
        prefetch_buffer=10,
        prefetch_buffer_keep=2,
        num_parts=kvstore.num_workers,
        part_index=kvstore.rank)

    return train, val


get_iterator.num_train_examples = 1281167