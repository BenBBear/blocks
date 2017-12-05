import mxnet as mx
import argparse
from utils.logger import get_logger
from utils.parse_args import parse_args, str2bool
from utils.fit import fit
from utils.test import test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast BNN Training Program")
    parser.add_argument("--symbol", type=str, default="symbol.resnet_alpha",
                        help="py module path to symbol (in symbol folder)")
    parser.add_argument("--initializer", type=str, default="initializer.default",
                        help="py module path to initializer (in initializer folder)")
    parser.add_argument("--iterator", type=str, default="iterator.cifar10",
                        help="py module path to iterator (in iterator folder)")
    parser.add_argument("--optimizer", type=str, default="optimizer.adam",
                        help="py module path to optimizer (in optimizer folder)")
    parser.add_argument("--metric", type=str, default="metric.acc",
                        help="py module path to metric (in metric folder)")
    parser.add_argument("--lr-scheduler", type=str, default="lr_scheduler.step",
                        help="py module path to lr_scheduler (in lr_scheduler folder)")
    parser.add_argument("--monitor", type=str, default="monitor.none",
                        help="py module path to monitor (in monitor folder)")
    parser.add_argument("--symbol-arg", type=str, default=None,
                        help="a=1,b=3 like param dict string, will pass to get_symbol")
    parser.add_argument("--optimizer-arg", type=str, default=None,
                        help="a=1,b=3 like param dict string, will pass to get_optimizer")
    parser.add_argument("--initializer-arg", type=str, default=None,
                        help="a=1,b=3 like param dict string, will pass to get_initializer")
    parser.add_argument("--metric-arg", type=str, default=None,
                        help="a=1,b=3 like param dict string, will pass to get_metric")
    parser.add_argument("--iterator-arg", type=str, default=None,
                        help="a=1,b=3 like param dict string, will pass to get_iterator")
    parser.add_argument("--lr-scheduler-arg", type=str, default=None,
                        help="a=1,b=3 like param dict string, will pass to get_lr_scheduler")
    parser.add_argument("--monitor-arg", type=str, default=None,
                        help="a=1,b=3 like param dict string, will pass to get_monitor")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="batch size")
    parser.add_argument("--gpus", type=str, default=None,
                        help="0,1,2 like string, indicate gpu num")
    parser.add_argument("--speed-meter", type=int, default=20,
                        help="train progress logging frequency")
    parser.add_argument("--num-epochs", type=int, default=120,
                        help="the total number of epochs")
    parser.add_argument("--save-model-prefix", type=str, default='./tmp/model',
                        help="prefix to save model checkpoint, usually hdfs url")
    parser.add_argument("--save-model-period", type=int, default=10,
                        help="model checkpoint period")
    parser.add_argument("--load-model-prefix", type=str, default=None,
                        help="load param from this prefix to initialize model before training")
    parser.add_argument("--load-epoch", type=int, default=0,
                        help="load model from this epoch")
    parser.add_argument("--begin-epoch", type=int, default=0,
                        help="begin epoch for training, (possibly work with steps in optimizer)")
    parser.add_argument("--log-dir", type=str, default="log",
                        help="log dir, usually set to hdfs filesystem uri")
    parser.add_argument("--log-file", type=str, default=None,
                        help="log file name, default set to [symbol_name].log")
    parser.add_argument("--allow-missing", type=str2bool, default=False,
                        help="whether allow missing params, when load param for model initialization")
    parser.add_argument("--test", type=str2bool, default=False,
                        help="whether run test program or train(fit) program")
    parser.add_argument("--freeze-pattern", type=str, default="",
                        help="regex pattern string, fix matched param when training")

    args = parse_args(parser)
    if args.test:
        test(args)
    else:
        fit(args)

