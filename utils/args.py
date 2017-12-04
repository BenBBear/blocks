

def get(kwargs, name, type=str, default=None):
    if name in kwargs:
        if type:
            r = type(kwargs[name])
        else:
            r = kwargs[name]
    else:
        r = default
    kwargs[name] = r
    return r


def _get_val(obj):
    if type(obj) is object:
        return vars(obj)
    else:
        return obj


def check(kwargs, scope, name_list):
    from logger import get_logger
    logger = get_logger()

    logger.info("Checking arguments for [{0}]".format(scope))
    for name in name_list:
        if name in kwargs:
            logger.info("\t{0} = {1}, type({2})", name, _get_val(kwargs[name]), type(kwargs[name]))
        else:
            logger.error("\t{0} does not exist!!")
