import mxnet as mx
from mxnet.tensorboard import MetricBoardCallback


def run_module(args):
    model = mx.mod.Module(symbol=args.sym_gen(), context=args.gpus, logger=args.logger,
                          data_names=[x[0] for x in args.iterator[1].provide_data],
                          label_names=[x[0] for x in args.iterator[1].provide_label],
                          fixed_param_names=args.fixed_param_names)

    model.bind(args.iterator[0].provide_data, args.iterator[0].provide_label)
    args.logger.info("=================== Begin Fit! ===================")
    model.fit(train_data=args.iterator[0], eval_data=args.iterator[1],
              monitor=args.monitor,
              eval_metric=args.eval_metric,
              validation_metric=args.validation_metric,
              kvstore=args.kvstore if args.gpus and len(args.gpus) > 1 else None,
              batch_end_callback=[mx.callback.Speedometer(args.batch_size, args.speed_meter),
                                  MetricBoardCallback(args.symbol_name + '.train')],
              eval_end_callback=[MetricBoardCallback(args.symbol_name + '.val'), ],
              epoch_end_callback=args.epoch_end_callback,
              num_epoch=args.num_epochs,
              begin_epoch=args.begin_epoch,
              optimizer=args.optimizer['optimizer'],
              optimizer_params=args.optimizer['optimizer_params'],
              initializer=args.initializer,
              arg_params=args.arg_params,
              aux_params=args.aux_params,
              allow_missing=args.allow_missing)
    args.logger.info("=================== End Fit! ===================")