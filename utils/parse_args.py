from importlib import import_module
from logger import get_logger
import argparse
from constant import RESCALE_GRAD
import mxnet as mx
import re


def str2list(st, tp=int):
    """
    convert a "a,b,c,d" like str to [a, b, c, d]
    :param st: original string
    :type st: str
    :param tp: conversion function
    :type tp: callable
    :return: result list
    :type: list
    """
    return [tp(x) for x in st.split(",")] if st else None


def str2bool(v):
    """
    convert a bool-indicated value into bool type
    :param v: bool-indicated value like "yes", "0"
    :type v: Union[str, bool, int]
    :return: bool value
    :type: bool
    """
    v = str(v)
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_str_arg(string):
    """
    convert a "a=1,b=3,c=4" like string into dict
    :param string: arg like string
    :type string: str
    :return: the result dict
    :type: dict
    """
    return {a[0]: a[1] for a in [
            s.split("=") for s in string.split(",")]} if string else {}


def parse_args(parser):
    """
    parse argument, will add/update these values like:
        - gpus
        - kvstore
        - epoch_size (from iterator.num_train_examples)
        - logger
        - symbol/monitor/optimizer/initializer/iterator/metric and .*_name
    :param parser: parser object created from `argparse.ArgumentParser()`
    :type parser: argparse.ArgumentParser
    :return: parsed argument
    :type: argparse.Namespace
    """
    args = parser.parse_args()
    logger = get_logger(args)
    args.logger = logger
    args.gpus = str2list(args.gpus)
    args.gpus = [mx.cpu(), ] if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus]
    args.symbol_arg = parse_str_arg(args.symbol_arg)
    args.optimizer_arg = parse_str_arg(args.optimizer_arg)
    args.initializer_arg = parse_str_arg(args.initializer_arg)
    args.metric_arg = parse_str_arg(args.metric_arg)
    args.iterator_arg = parse_str_arg(args.iterator_arg)
    args.lr_scheduler_arg = parse_str_arg(args.lr_scheduler_arg)
    args.monitor_arg = parse_str_arg(args.monitor_arg)

    if len(args.gpus) >= 2:
        kv_store = 'device'
    else:
        kv_store = 'local'
    kv = mx.kvstore.create(kv_store)
    args.kvstore = kv

    args.initializer_name = args.initializer
    args.initializer = import_module(args.initializer).get_initializer
    args.iterator_name = args.iterator
    args.iterator = import_module(args.iterator).get_iterator
    args.metric_name = args.metric
    args.metric = import_module(args.metric).get_metric
    args.optimizer_name = args.optimizer
    args.optimizer = import_module(args.optimizer).get_optimizer
    args.lr_scheduler_name = args.lr_scheduler
    args.lr_scheduler = import_module(args.lr_scheduler).get_lr_scheduler
    args.symbol_name = args.symbol
    args.symbol = import_module(args.symbol).get_symbol
    if args.monitor:
        args.monitor = import_module(args.monitor).get_monitor

    args.epoch_size = max(int(args.iterator.num_train_examples / args.batch_size / kv.num_workers), 1)

    optimizer_batch_size = args.batch_size
    if kv and ('dist' in kv.type) and ('_async' not in kv.type):
        optimizer_batch_size *= kv.num_workers

    if 'rescale_grad' not in args.optimizer['optimizer_params']:
        args.optimizer['optimizer_params']['rescale_grad'] = RESCALE_GRAD['divide_batch_size']

    rescale_grad = args.optimizer['optimizer_params']['rescale_grad']
    if rescale_grad == RESCALE_GRAD['divide_batch_size']:
        args.optimizer['optimizer_params']['rescale_grad'] = 1.0 / optimizer_batch_size
    elif rescale_grad == RESCALE_GRAD['divide_num_devs']:
        args.optimizer['optimizer_params']['rescale_grad'] = 1.0 / len(args.gpus)
    else:
        raise Exception("Unrecognized rescale_grad option " + args.optimizer['optimizer_params']['rescale_grad'])

    logger.info('start with arguments %s', vars(args))
    return args
