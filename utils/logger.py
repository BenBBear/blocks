
def get_logger(args=None):
    if get_logger.logger is not None:
        return get_logger.logger

    import logging
    from os.path import exists, join
    from os import mkdir
    head = '%(asctime)-15s Node[' + str(0) + '] %(message)s'
    formatter = logging.Formatter(head)
    logger = logging.getLogger()
    try:
        if args.log_dir and (not exists(args.log_dir)):
            mkdir(args.log_dir)
        if not args.log_file:
            args.log_file = args.iterator.split('.')[-1] + '.' + args.symbol.split('.')[-1] + '.txt'
        handler = logging.FileHandler(join(args.log_dir, args.log_file))
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    except Exception:
        pass

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    get_logger.logger = logger
    return logger

get_logger.logger = None