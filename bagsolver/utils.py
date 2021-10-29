from functools import wraps
import time


def load_bag_data(x_file, y_file):
    with open(x_file) as x, open(y_file) as y:
        # Deduplicate multiple solutions in y
        xdict = {line.split(" ")[0]: line for line in x.readlines()}
        ydict = {line.split(" ")[0]: line for line in y.readlines()}
        xlines = (v for (k, v) in sorted(xdict.items(), key=lambda x: x[0]))
        ylines = (v for (k, v) in sorted(ydict.items(), key=lambda x: x[0]))
        return tuple(zip(xlines, ylines))


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
