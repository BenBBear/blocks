import mxnet as mx
import argparse
from utils.logger import get_logger
from utils.parse_args import parse_args, str2bool
from utils.fit import fit
from utils.test import test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast BNN Training Program")
    parser.add_argument("--symbol", type=str, default="symbol.resnet_alpha")
    parser.add_argument("--initializer", type=str, default="initializer.default")
    parser.add_argument("--iterator", type=str, default="iterator.cifar10")
    parser.add_argument("--optimizer", type=str, default="optimizer.adam")
    parser.add_argument("--metric", type=str, default="metric.acc")
    parser.add_argument("--lr-scheduler", type=str, default="lr_scheduler.step")

    parser.add_argument("--symbol-arg", type=str, default=None)
    parser.add_argument("--optimizer-arg", type=str, default=None)
    parser.add_argument("--initializer-arg", type=str, default=None)
    parser.add_argument("--metric-arg", type=str, default=None)
    parser.add_argument("--iterator-arg", type=str, default=None)
    parser.add_argument("--lr-scheduler-arg", type=str, default=None)

    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--gpus", type=str, default=None)

    parser.add_argument("--speed-meter", type=int, default=20)
    parser.add_argument("--num-epochs", type=int, default=120)

    parser.add_argument("--save-model-prefix", type=str, default='./tmp/model')
    parser.add_argument("--save-model-period", type=int, default=10)
    parser.add_argument("--load-model-prefix", type=str, default=None)
    parser.add_argument("--load-epoch", type=int, default=0)
    parser.add_argument("--begin-epoch", type=int, default=0)

    parser.add_argument("--log-dir", type=str, default="log")
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--monitor", type=str, default="monitor.none")
    parser.add_argument("--monitor-arg", type=str, default=None)
    parser.add_argument("--allow-missing", type=str2bool, default=False)
    parser.add_argument("--test", type=str2bool, default=False)
    parser.add_argument("--freeze-pattern", type=str, default="")

    args = parse_args(parser)
    if args.test:
        test(args)
    else:
        fit(args)

