import timeit

default_template = timeit.template

# new template allowing to retrieve return values function from timeit
template = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        ret_val = {stmt}
    _t1 = _timer()
    return _t1 - _t0, ret_val
"""

timeit.template = template


def set_default():
    timeit.template = default_template


def set_template():
    timeit.template = template


# def run_example():
#     time, return_values = timeit.timeit(stmt="run()", setup="from __main__ import run", number=1)
