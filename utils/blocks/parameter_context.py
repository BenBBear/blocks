from collections import deque
from ..args import get


class ParameterContext(object):
    stack = deque()

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

    def __exit__(self, exc_type, exc_val, exc_tb):
        ParameterContext.stack.pop()


def cget(kwargs, key, func, default):
    return get(kwargs, key, func, ParameterContext.get(key, default))

context_get = cget