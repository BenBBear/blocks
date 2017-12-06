from op_block import *
context = cget


def block(name=None, **kwargs):
    return OpBlock(name=name, **kwargs)


def var(name=None, data_iter=None, **kwargs):
    return OpBlock(name=name, data_iter=data_iter, **kwargs)


def group(blk_list, **kwargs):
    return GroupBlock(block_list=blk_list, **kwargs)


def conv2d(data, num_filter=1, kernel=3, stride=1, pad=None, dilate=1, num_group=1, **kwargs):
    blk = ConvBlock(**kwargs)
    blk(data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, dilate=dilate,
        num_group=num_group, **kwargs)
    return blk


def avg_pool2d(data, kernel=3, stride=2, pad=None, **kwargs):
    blk = PoolBlock(**kwargs)
    blk(data, kernel=kernel, stride=stride, pad=pad, pool_type='avg')
    return blk


def max_pool2d(data, kernel=3, stride=2, pad=None, **kwargs):
    blk = PoolBlock(**kwargs)
    blk(data, kernel=kernel, stride=stride, pad=pad, pool_type='max')
    return blk


def global_pool2d(data, **kwargs):
    blk = PoolBlock(**kwargs)
    blk(data, kernel=1, stride=1, pad=0, pool_type='avg', global_pool=True, **kwargs)
    return blk


def batch_norm(data, **kwargs):
    blk = BatchNormBlock(**kwargs)
    blk(data, **kwargs)
    return blk


def dropout(data, prob, **kwargs):
    blk = DropoutBlock(**kwargs)
    blk(data, prob=prob, **kwargs)
    return blk


def dropcell(data, prob, **kwargs):
    blk = DropCellBlock(**kwargs)
    blk(data, prob=prob, **kwargs)
    return blk


def fully_connected(data, num_hidden, **kwargs):
    blk = DenseBlock(**kwargs)
    blk(data, num_hidden=num_hidden, **kwargs)
    return blk


def conv2d_bn(data, *args, **kwargs):
    return batch_norm(conv2d(data, *args, **kwargs), **kwargs)


def sconv2d_bn(data, num_filter=1, kernel=3, stride=1, pad=None, dilate=1, num_group=1, **kwargs):
    # depthwise -> pointwise -> bn
    s_num_group = data.shape[data.channel_id]
    blk1 = conv2d(data, s_num_group, kernel, stride, pad=pad, dilate=dilate, num_group=s_num_group, **kwargs)
    blk2 = conv2d(blk1, num_filter, 1, 1, pad=0, dilate=1, num_group=1, **kwargs)
    blk3 = batch_norm(blk2, **kwargs)
    return blk3