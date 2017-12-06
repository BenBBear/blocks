from ..args import get
import mxnet as mx
from ..base import num_as_tuple
from contexts.parameter_context import cget, context_set, Context
from collections import deque, defaultdict


class ModeSwitch(object):
    """
    trait that enable OpBlock to behave differently when chaining together
    """
    mode_train = 'train'
    mode_inference = 'inference'
    # mode_inference_extra = '8bit/alpha/beta'

    def __init__(self, **kwargs):
        self.mode = cget(kwargs, 'mode', str, 'train')

    def train(self, *args, **kwargs):
        """
        when self.mode == "train", run this
        """
        raise NotImplementedError

    def inference(self, *args, **kwargs):
        """
        otherwise, it will run inference chain operator (and of course it may have different inference behavior)
        """
        self.train(*args, **kwargs)


class WeightBias(ModeSwitch):
    """
    trait that patch a block with weight/bias/no_bias parameter
    """
    def __init__(self, **kwargs):
        ModeSwitch.__init__(**kwargs)
        self.weight = None
        self.bias = None
        self.no_bias = False

    def get_weight_bias(self, **kwargs):
        weight_initializer = cget(kwargs, 'weight_initializer', lambda x: x, None)
        bias_initializer = cget(kwargs, 'bias_initializer', lambda x: x, None)
        self.no_bias = cget(kwargs, 'no_bias', bool, False)
        self.weight = mx.sym.var(self.name + '_weight', init=weight_initializer)
        self.bias = mx.sym.var(self.name + '_bias', init=bias_initializer) if not self.no_bias else None
        return self.weight, self.bias, self.no_bias


class CounterReset(object):
    """
    trait that can reset counter when used under with Context(reset_counter=True)
    """
    root_counter = defaultdict(lambda: 0)
    counter_stack = deque([root_counter, ], )

    @staticmethod
    def create_counter(self):
        return defaultdict(lambda: 0)

    @property
    def counter(self):
        return self.counter_stack[-1]

    @staticmethod
    def push_counter():
        p = Context.current()
        if 'reset_counter' in p and p['reset_counter'] is True:
            CounterReset.counter_stack.append(CounterReset.create_counter())

    @staticmethod
    def pop_counter():
        p = Context.current()
        if 'reset_counter' in p and p['reset_counter'] is True:
            CounterReset.counter_stack.pop()


Context.bind('enter', lambda: CounterReset.push_counter())
Context.bind('exit', lambda: CounterReset.pop_counter())


class Trait(object):
    def parse_args(self, *args, **kwargs):
        raise NotImplementedError


class ConvTrait(Trait):
    """
    trait that patch a block with convolution parameter
    """
    def __init__(self, **kwargs):
        self.kernel = ()
        self.stride = ()
        self.pad = ()
        self.num_group = 1
        self.dilate = ()
        self.num_filter = 1

    def parse_args(self, **kwargs):
        self.kernel = num_as_tuple(cget(kwargs, "kernel", tuple, (1, 1)))
        self.stride = num_as_tuple(cget(kwargs, "stride", tuple, (1, 1)))
        self.dilate = num_as_tuple(cget(kwargs, "dilate", tuple, (1, 1)))
        self.pad = num_as_tuple(cget(kwargs, "pad", tuple, (int(self.kernel[0]/2), int(self.kernel[1]/2))))
        self.num_group = cget(kwargs, "num_group", int, 1)
        self.num_filter = cget(kwargs, "num_filter", int, 1)


class PoolTrait(Trait):
    """
    trait that patch a block with pooling parameter
    """
    def __init__(self, **kwargs):
        self.kernel = ()
        self.stride = ()
        self.pad = ()
        self.global_pool = False
        self.pool_type = ''  # max, avg, global_avg

    def parse_args(self, **kwargs):
        self.kernel = num_as_tuple(cget(kwargs, "kernel", tuple, (1, 1)))
        self.stride = num_as_tuple(cget(kwargs, "stride", tuple,  (1, 1)))
        self.pad = num_as_tuple(cget(kwargs, "pad", tuple, (int(self.kernel[0]/2), int(self.kernel[1]/2))))
        self.pool_type = cget(kwargs, "pool_type", str, 'max')
        self.global_pool = cget(kwargs, "global_pool", bool, False)


class DropTrait(Trait):
    def __init__(self, **kwargs):
        self.prob = 0.0

    def parse_args(self, **kwargs):
        self.prob = cget(kwargs, "prob", float, 0.0)


class BatchNormTrait(Trait):
    def __init__(self, **kwargs):
        self.eps = 2e-5
        self.fix_gamma = True,
        self.momentum = .9
        self.use_global_stats = False

    def parse_args(self, **kwargs):
        self.eps = cget(kwargs, "eps", float, 2e-5)
        self.fix_gamma = cget(kwargs, "fix_gamma", bool, True)
        self.momentum = cget(kwargs, "momentum", float, .9)
        self.use_global_stats = cget(kwargs, "use_global_stats", bool, False)


class LayoutIndicator(object):
    def __init__(self, **kwargs):
        self.layout = cget(kwargs, "layout", str, "NCHW")
        self.channel_id = self.layout.find("C")
        self.height_id = self.layout.find("H")
        self.width_id = self.layout.find("W")
        self.time_id = self.layout.find("T")
        self.batch_id = self.layout.find("N")
