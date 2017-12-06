from ..harden.metric import HardwareMetric, HardwareMetricContainer
from traits import *
import re


class Block(object):
    """
    building block for compose mxnet symbol, the design philosophy are:
        - Chain of blocks (see __call__)
        - Hierarchy of blocks (see self.inner_blocks, _stacking, prev_blocks)
    """
    def __init__(self, name=""):
        """
        create a empty Block instance
        :param name: the name of this block
        """
        if name.count("::") > 0:
            raise Exception("Block name {0} is illegal, \"::\" should not be used!".format(name))
        self.level = 0
        self.name = name
        self.symbol = None

        # for hierarchical management and processing (like ops calculation, micro-architecture comparision)
        self._stacking = False
        self.inner_blocks = None
        # prev_block can be Block, list of Block, dict of str,block
        self.prev_block = None

    def __enter__(self):
        """
        entering stacking mode, to build hierarchy for blocks
        (this block represent no symbol, but a collection of blocks)
        :return: None
        """
        assert self.symbol is None, "you cannot stack into a block with symbol"
        self._stacking = True
        # current block is self.inner_blocks[-1]
        self.inner_blocks = []
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        exit stacking mode
        :return: None
        """
        self._stacking = False

    def add(self, block):
        """
        stack block into this block(container)
        :param block: blk instance
        :type block: Block
        :return: self
        """
        assert self._stacking, "block can append inner_blocks only in with-statement."
        if block.level < self.name.count("::") + 1:
            block.name = self.name + "::" + block.name
            block.level = block.name.count("::")
        self.inner_blocks.append(block)
        self.symbol = block.symbol
        return self

    def __call__(self, block, *args, **kwargs):
        """
        chaining operation
        """
        self.prev_block = block


class BlockWithShape(Block):
    """
    block class that can has shape property
    """
    def __init__(self, data_iter=None, name=""):
        """
        create a BlockWithShape Instance
        :param data_iter: data iterator (need its provide_data/label)
        :type data_iter: mx.io.DataIter
        :param name: name of the block
        """
        super(BlockWithShape, self).__init__(name=name)
        # argument pass to symbol.infer_shape
        self.__shape_dict = {}
        if data_iter:
            for provide in [data_iter.provide_data, data_iter.provide_label]:
                if provide:
                    for name, shape in provide:
                        self.__shape_dict[name] = shape

    def add(self, block):
        """
        same of Block.add, except it will pass the block.__shape_dict to this container block
        """
        assert isinstance(block, BlockWithShape), "block {0} does not have shape information".format(block.name)
        super(BlockWithShape, self).add(block)
        # update shape dict, because the current block is changed
        self.__shape_dict = block.__shape_dict

    @property
    def shape(self):
        """
        shape getter method
        :return: Union[tuple, list of tuple]
        """
        arg_shape, out_shape, aux_shape = self.symbol.infer_shape(**self.__shape_dict)
        if len(out_shape) == 1:
            return out_shape[0]
        else:
            return out_shape

    def __call__(self, block, *args, **kwargs):
        """
        when it chains with another block, propagate the shape information
        """
        super(BlockWithShape, self).__call__(block)
        self.__shape_dict = block.__shape_dict

    def check_shape(self):
        """
        check shape for this block, usually used in __call__, it will raise Exception if failed.
        """
        _ = self.symbol.infer_shape(**self.__shape_dict)
        return True


class BlockWithCalOps(BlockWithShape):
    """
    block that support hardware resource estimation on this Block (including its inner block)
    """
    def __init__(self, **kwargs):
        super(BlockWithCalOps, self).__init__(**kwargs)
        # set to True if this block is already counted

    def ops(self, result=None, max_level=-1, regex=".*", _level=0):
        result = HardwareMetricContainer() if result is None else result
        regex = re.compile(regex) if isinstance(regex, str) else regex
        _level = _level + 1
        if max_level > 0:
            if _level > max_level:
                return
        if self.inner_blocks and len(self.inner_blocks) > 0:
            for b in self.inner_blocks:
                b.ops(result=result, max_level=max_level, _level=_level, regex=regex)
            result.clean()
            return result
        else:
            _ops = self.cal_ops()
            _ops.name = self.name
            if regex.match(self.name) and _ops is not None:
                result.append(_ops)
            return result

    def cal_ops(self):
        """
        cal_ops should return a HardwareMetric instance
        :return: HardwareMetric instance
        """
        return None

    def show(self, max_level=-1, regex=".*"):
        return self.ops(result=None, max_level=max_level, regex=regex, _level=0)


class OpBlock(BlockWithCalOps, ModeSwitch):
    counter = 0
    prefix = 'block'

    def __init__(self, name=None, **kwargs):
        if not name:
            name = self.prefix + str(self.counter)
            self.counter += 1
        super(OpBlock, self).__init__(name=name, **kwargs)
        ModeSwitch.__init__(**kwargs)

    def __call__(self, block, *args, **kwargs):
        BlockWithCalOps.__call__(block, *args, **kwargs)
        if self.mode == self.mode_train:
            self.symbol = self.train(*args, **kwargs)
        else:
            self.symbol = self.inference(*args, **kwargs)
        self.check_shape()