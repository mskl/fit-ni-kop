from functools import wraps
from typing import Dict, NamedTuple
import numpy as np
import time


def timed(f):
    @wraps(f)
    def wrap(*args, **kw):
        started = time.time()
        result = f(*args, **kw)
        elapsed = time.time() - started
        return (elapsed, result)
    return wrap


class Solution(NamedTuple):
    optscore: int
    solution: np.ndarray


def load_solutions(filepath: str) -> Dict[str, Solution]:
    solutions = {}
    with open(filepath) as fp:
        for line in fp.readlines():
            name, optscore, *solution = line.split(" ")
            parsed_solution = np.sign([int(_) for _ in solution[:-1]])
            solutions[name] = Solution(int(optscore), parsed_solution)
    return solutions

