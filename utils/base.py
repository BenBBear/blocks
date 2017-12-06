

def unique_list(lst, key=lambda x: x):
    """
    remove redundant items in a list
    """
    d = {}
    for v in lst:
        d[key(v)] = d
    return [v for v in d.values()]


def prod_list(lst):
    """
    calculate the multiply of all items in a list
    """
    p = 1
    for i in lst:
        p *= i
    return p


def num_as_tuple(num, size=2):
    if isinstance(num, tuple) or isinstance(num, list):
        return tuple(num)
    else:
        return tuple([num, ] * size)