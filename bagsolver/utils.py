from functools import wraps
import time


def load_bag_data(x_file, y_file):
    with open(x_file) as x, open(y_file) as y:
        return tuple(zip(x.readlines(), y.readlines()))


def parse_solution(bag_sol):
    parsed = [int(v) for v in bag_sol.strip().split(" ")]
    iid, count, target_cost, *target_items = parsed
    return iid, count, target_cost, target_items


def timed(f):
    @wraps(f)
    def wrap(*args, **kw):
        started = time.time()
        result = f(*args, **kw)
        elapsed = time.time() - started
        return (elapsed, result)
    return wrap
