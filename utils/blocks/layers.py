from op_block import *
context = cget


def block(name=None):
    return OpBlock(name=name)


def var(name=None, data_iter=None):
    return OpBlock(name=name, data_iter=data_iter)


def group(blk_list, name=None):
    return GroupBlock(block_list=blk_list, name=name)


def conv2d(data, **kwargs):
    pass


def avg_pool2d(data, **kwargs):
    pass


def max_pool2d(data, **kwargs):
    pass


def global_pool2d(data, **kwargs):
    pass


def batch_norm(data, **kwargs):
    pass


def dropout(data, **kwargs):
    pass


def dropcell(data, **kwargs):
    pass


def fully_connected(data, **kwargs):
    pass


def conv2d_bn(data, **kwargs):
    pass


def sconv2d_bn(data, **kwargs):
    pass