from collections import deque


class ParameterContext(object):
    stack = deque()
    _event_stack = {'enter': [], 'exit': []}

    @staticmethod
    def current():
        if len(ParameterContext.stack) > 0:
            return ParameterContext.stack[-1]
        else:
            return None

    @staticmethod
    def get(key, default):
        current = ParameterContext.current()
        if current and key in current:
            return current[key]
        else:
            return default

    def __init__(self, **kwargs):
        self.parameter = kwargs

    def __enter__(self):
        self.parameter.update(ParameterContext.current())
        ParameterContext.stack.append(self.parameter)
        for f in self._event_stack['enter']:
            f(self.parameter)

    def __exit__(self, exc_type, exc_val, exc_tb):
        p = ParameterContext.stack.pop()
        for f in self._event_stack['exit']:
            f(p)

    @staticmethod
    def size():
        return len(ParameterContext.stack)

    @staticmethod
    def bind(name, callback):
        ParameterContext._event_stack[name].append(callback)


Context = ParameterContext


def cget(kwargs, name, type=str, default=None):
    """
    get a value from kwargs/context using key(name), with type conversion and default value
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
        if kwargs[name] is None:
            # read from context, if this param is None
            r = ParameterContext.get(name, default)
        else:
            # else read from function arguments
            if type:
                r = type(kwargs[name])
            else:
                r = kwargs[name]
    else:
        # if this param is not given, read from context
        r = ParameterContext.get(name, default)
    return r


context_get = cget


def cset(key, value):
    current = ParameterContext.current()
    current[key] = value


context_set = cset

