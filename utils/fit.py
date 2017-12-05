import mxnet as mx
from mxnet.tensorboard import MetricBoardCallback
import re


def fit(args):
    """
    train model based on args provided
    :param args: args return from parse_args
    :type args: argparse.Namespace
    :return: None
    """
    logger = args.logger

    def sym_gen():
        return args.symbol(**args.symbol_arg)
    symbol = sym_gen()
    initializer = args.initializer(console_arg=args, **args.initializer_arg)
    iterator = args.iterator(batch_size=args.batch_size, console_arg=args, kvstore=args.kvstore,
                             **args.iterator_arg)
    optimizer = args.optimizer(console_arg=args, **args.optimizer_arg)
    lr_scheduler = args.lr_scheduler(console_arg=args, **args.lr_scheduler_arg)
    optimizer['optimizer_params']['lr_scheduler'] = lr_scheduler
    metric = args.metric(console_arg=args, **args.metric_arg)

    if isinstance(metric, dict):
        eval_metric = metric['eval_metric']
        validation_metric = metric['validation_metric']
    else:
        eval_metric = metric
        validation_metric = metric

    arg_params, aux_params = None, None
    if args.load_model_prefix:
        logger.warn("Loading checkpoint from {0}, at epoch {1}".format(
            args.load_model_prefix, args.load_epoch))
        _, arg_params, aux_params = mx.model.load_checkpoint(
            args.load_model_prefix, args.load_epoch)
        logger.warn("Done checkpoint loading")

    epoch_end_callback = []
    if args.save_model_prefix:
        checkpoint = mx.callback.do_checkpoint(
            args.save_model_prefix, period=args.save_model_period)
        epoch_end_callback.append(checkpoint)

    if args.freeze_pattern.strip():
        re_prog = re.compile(args.freeze_pattern)
        fixed_param_names = [
            name for name in symbol.list_arguments() if re_prog.match(name)
        ]
    else:
        fixed_param_names = None
    if fixed_param_names:
        logger.info("Fixed parameters: [" + ','.join(fixed_param_names) + ']')

    model = mx.mod.Module(symbol=symbol, context=args.gpus, logger=logger,
                          data_names=[x[0] for x in iterator[1].provide_data],
                          label_names=[x[0] for x in iterator[1].provide_label],
                          fixed_param_names=fixed_param_names)

    monitor = None
    if args.monitor:
        monitor = args.monitor(console_arg=args, **args.monitor_arg)

    model.bind(iterator[0].provide_data, iterator[0].provide_label)
    logger.info("=================== Begin Fit! ===================")
    model.fit(train_data=iterator[0], eval_data=iterator[1],
              monitor=monitor,
              eval_metric=eval_metric,
              validation_metric=validation_metric,
              kvstore=args.kvstore if args.gpus and len(args.gpus) > 1 else None,
              batch_end_callback=[mx.callback.Speedometer(args.batch_size, args.speed_meter),
                                  MetricBoardCallback(args.symbol_name + '.train')],
              eval_end_callback=[MetricBoardCallback(args.symbol_name + '.val'), ],
              epoch_end_callback=epoch_end_callback,
              num_epoch=args.num_epochs,
              begin_epoch=args.begin_epoch,
              optimizer=optimizer['optimizer'],
              optimizer_params=optimizer['optimizer_params'],
              initializer=initializer,
              arg_params=arg_params,
              aux_params=aux_params,
              allow_missing=args.allow_missing)
    logger.info("=================== End Fit! ===================")