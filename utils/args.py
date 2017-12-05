

def get(kwargs, name, type=str, default=None):
    """
    get a value from kwargs using key(name), with type conversion and default value
    :param kwargs: parameter dictionary
    :type kwargs: dict
    :param name: parameter name
    :type name: str
    :param type: type conversion function
    :type type: callable
    :param default: default parameter
    :type default: any
    :return: the corespondent value
    :type: any
    """
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
    """
    parse a object into a printable dict, if not object then return as it
    :param obj: object
    :type: any
    :return: printable value
    """
    if type(obj) is object:
        return vars(obj)
    else:
        return obj


def check(kwargs, scope, name_list):
    """
    check existence of the parameter name from name_list
    :param kwargs: parameter dict
    :type kwargs: dict
    :param scope: string to be printed, indicate where the check is happening
    :type scope: str
    :param name_list: parameter name list
    :param name_list: list of str
    :return: None
    """
    from logger import get_logger
    logger = get_logger()

    logger.info("Checking arguments for [{0}]".format(scope))
    for name in name_list:
        if name in kwargs:
            logger.info("\t{0} = {1}, type({2})", name, _get_val(kwargs[name]), type(kwargs[name]))
        else:
            logger.error("\t{0} does not exist!!")
