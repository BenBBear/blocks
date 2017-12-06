import mxnet as mx
from mxnet.tensorboard import MetricBoardCallback
import re
from utils.blocks.contexts.parameter_context import Context


def fit(args):
    """
    train model based on args provided
    :param args: args return from parse_args
    :type args: argparse.Namespace
    :return: None
    """
    logger = args.logger

    def sym_gen():
        mode = 'train'
        if 'mode' in args.symbol_arg:
            mode = args.symbol_arg['mode']
        with Context(reset_counter=True, mode=mode):
            return args.symbol(**args.symbol_arg)
    symbol = sym_gen()
    args.sym_gen = sym_gen
    args.initializer = args.initializer(console_arg=args, **args.initializer_arg)
    args.iterator = iterator = args.iterator(batch_size=args.batch_size, console_arg=args,
                                             kvstore=args.kvstore, **args.iterator_arg)
    optimizer = args.optimizer(console_arg=args, **args.optimizer_arg)
    args.lr_scheduler = args.lr_scheduler(console_arg=args, **args.lr_scheduler_arg)
    optimizer['optimizer_params']['lr_scheduler'] = args.lr_scheduler
    args.optimizer = optimizer
    args.metric = metric = args.metric(console_arg=args, **args.metric_arg)

    if isinstance(metric, dict):
        eval_metric = metric['eval_metric']
        validation_metric = metric['validation_metric']
    else:
        eval_metric = metric
        validation_metric = metric
    args.eval_metric = eval_metric
    args.validation_metric = validation_metric

    arg_params, aux_params = None, None
    if args.load_model_prefix:
        logger.warn("Loading checkpoint from {0}, at epoch {1}".format(
            args.load_model_prefix, args.load_epoch))
        _, arg_params, aux_params = mx.model.load_checkpoint(
            args.load_model_prefix, args.load_epoch)
        logger.warn("Done checkpoint loading")
    args.arg_params = arg_params
    args.aux_params = aux_params

    epoch_end_callback = []
    if args.save_model_prefix:
        checkpoint = mx.callback.do_checkpoint(
            args.save_model_prefix, period=args.save_model_period)
        epoch_end_callback.append(checkpoint)
    args.epoch_end_callback = epoch_end_callback

    if args.freeze_pattern.strip():
        re_prog = re.compile(args.freeze_pattern)
        fixed_param_names = [
            name for name in symbol.list_arguments() if re_prog.match(name)
        ]
    else:
        fixed_param_names = None
    if fixed_param_names:
        logger.info("Fixed parameters: [" + ','.join(fixed_param_names) + ']')
    args.fixed_param_names = fixed_param_names
    args.context = args.gpus
    args.batch_end_callback = [mx.callback.Speedometer(args.batch_size, args.speed_meter),
                               MetricBoardCallback(args.symbol_name + '.train')]
    args.eval_end_callback = [MetricBoardCallback(args.symbol_name + '.val'), ]
    args.bind_args = (iterator[0].provide_data, iterator[0].provide_label)
    args.data_names, args.label_names = [x[0] for x in iterator[1].provide_data], \
                                        [x[0] for x in iterator[1].provide_label]

    if args.monitor:
        args.monitor = monitor = args.monitor(console_arg=args, **args.monitor_arg)

    args.run_module(**vars(args))

