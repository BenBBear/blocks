import mxnet as mx
from block import *
from traits import *
from ..base import prod_list
from ..harden.metric import FLOAT_SIZE

# TODO, the cal_ops function defined here, doesn't consider layout (NHWC, NCHW)


class GroupBlock(OpBlock):
    prefix = 'group'

    def __init__(self, block_list, name=None, **kwargs):
        super(GroupBlock, self).__init__(name=name, **kwargs)
        self.inner_blocks = [b for b in block_list]
        self._group = True
        self.symbol = mx.sym.Group([b.symbol for b in self.inner_blocks])


class ConvBlock(OpBlock, WeightBias, ConvTrait):
    prefix = 'conv'

    def __init__(self, *args, **kwargs):
        OpBlock.__init__(*args, **kwargs)
        ConvTrait.__init__(*args, **kwargs)
        WeightBias.__init__(*args, **kwargs)

    def train(self, block, **kwargs):
        self.get_weight_bias(**kwargs)
        self.parse_args(**kwargs)
        return mx.sym.Convolution(data=block.symbol, weight=self.weight, bias=self.bias, no_bias=self.no_bias,
                                  kernel=self.kernel, stride=self.stride, pad=self.pad, dilate=self.dilate,
                                  num_group=self.num_group, name=self.name, num_filter=self.num_filter)

    def cal_ops(self):
        in_shape = self.prev_block.shape
        out_shape = self.shape
        in_filter = in_shape[self.channel_id]
        return HardwareMetric(name=self.name,
                              mac=in_filter * self.num_filter * out_shape[self.height_id] * out_shape[self.width_id],
                              input_size=prod_list(in_shape[1:]) * FLOAT_SIZE,
                              param_size=prod_list(self.kernel) * self.num_filter * in_filter * FLOAT_SIZE,
                              output_size=prod_list(out_shape[1:]) * FLOAT_SIZE)


class PoolBlock(OpBlock, PoolTrait):
    prefix = 'pool'

    def __init__(self, *args, **kwargs):
        OpBlock.__init__(*args, **kwargs)
        PoolTrait.__init__(*args, **kwargs)

    def train(self, block, **kwargs):
        self.parse_args(**kwargs)
        return mx.sym.Pooling(data=block.symbol, kernel=self.kernel, stride=self.stride, pad=self.pad,
                              name=self.name, pool_type=self.pool_type, global_pool=self.global_pool)

    def cal_ops(self):
        in_shape = self.prev_block.shape
        out_shape = self.shape
        return HardwareMetric(name=self.name,
                              input_size=prod_list(in_shape[1:]) * FLOAT_SIZE,
                              output_size=prod_list(out_shape[1:]) * FLOAT_SIZE)


class DenseBlock(OpBlock, WeightBias):
    prefix = 'dense'

    def __init__(self, **kwargs):
        OpBlock.__init__(**kwargs)
        WeightBias.__init__(**kwargs)
        self.num_hidden = 0

    def train(self, block, num_hidden=0, **kwargs):
        self.num_hidden = num_hidden
        self.get_weight_bias(**kwargs)
        return mx.sym.FullyConnected(data=block.symbol, num_hidden=num_hidden, weight=self.weight, bias=self.bias,
                                     no_bias=self.no_bias, name=self.name)

    def cal_ops(self):
        in_shape = self.prev_block.shape
        out_shape = self.shape
        in_hidden = in_shape[1]
        return HardwareMetric(name=self.name,
                              mac=self.num_hidden * in_hidden,
                              param_size=self.num_hidden * in_hidden * FLOAT_SIZE,
                              input_size=prod_list(in_shape[1:]) * FLOAT_SIZE,
                              output_size=prod_list(out_shape[1:]) * FLOAT_SIZE)


class DropoutBlock(OpBlock, DropTrait):
    prefix = 'dropout'

    def __init__(self, **kwargs):
        OpBlock.__init__(**kwargs)
        DropTrait.__init__(**kwargs)

    def train(self, block, **kwargs):
        return mx.sym.Dropout(block.symbol, p=self.prob, name=self.name)


class DropCellBlock(OpBlock, DropTrait):
    prefix = 'dropcell'

    def __init__(self, **kwargs):
        OpBlock.__init__(**kwargs)
        DropTrait.__init__(**kwargs)

    def train(self, block, **kwargs):
        self.parse_args(**kwargs)
        prev_shape = block.shape
        batch_size = prev_shape[0]
        switch = mx.sym.broadcast_to(data=mx.sym.uniform(low=0, high=1, shape=(1, )), shape=(batch_size, ))
        return mx.sym.where(switch > self.prob, x=block.symbol, y=mx.symbol.zeros(prev_shape), name=self.name)

    def inference(self, block, **kwargs):
        self.parse_args(**kwargs)
        return block.symbol * self.prob


class BatchNormBlock(OpBlock, BatchNormTrait):
    prefix = 'batch_norm'

    def __init__(self, **kwargs):
        OpBlock.__init__(**kwargs)
        BatchNormTrait.__init__(**kwargs)

    def train(self, block, **kwargs):
        self.parse_args(**kwargs)
        return mx.sym.BatchNorm(data=block.symbol, eps=self.eps, fix_gamma=self.fix_gamma,
                                use_global_stats=self.use_global_stats, momentum=self.momentum)

    def cal_ops(self):
        in_shape = self.prev_block.shape
        out_shape = self.shape
        if len(in_shape) > self.channel_id:
            in_channel = in_shape[self.channel_id]
        else:
            in_channel = in_shape[1]
        return HardwareMetric(name=self.name,
                              mac=prod_list(in_shape[1:]),
                              param_size=in_channel * FLOAT_SIZE * 2,
                              input_size=prod_list(in_shape[1:]) * FLOAT_SIZE,
                              output_size=prod_list(out_shape[1:]) * FLOAT_SIZE)


class AddBlock(OpBlock):
    prefix = 'add'
    elementwise_op = mx.sym._internal._plus

    def train(self, block1, block2, **kwargs):
        self.prev_block = {'left': block1, 'right': block2}
        return self.elementwise_op(block1.symbol, block2.symbol, name=self.name)


class SubBlock(AddBlock):
    prefix = 'sub'
    elementwise_op = mx.sym._internal._sub


class MulBlock(AddBlock):
    prefix = 'mul'
    elementwise_op = mx.sym._internal._mul

    def cal_ops(self):
        in_shape = self.prev_block.shape
        out_shape = self.shape
        return HardwareMetric(name=self.name,
                              mac=prod_list(in_shape[1:]),
                              param_size=0,
                              input_size=prod_list(in_shape[1:]) * FLOAT_SIZE,
                              output_size=prod_list(out_shape[1:]) * FLOAT_SIZE)


class DivBlock(MulBlock):
    prefix = 'div'
    elementwise_op = mx.sym._internal._div
