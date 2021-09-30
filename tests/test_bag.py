import os
from pathlib import Path

import pytest
from bagsolver.bag import Bag

THIS_DIR = Path(__file__).parent


@pytest.mark.parametrize(
    "rawline, expected",
    [
        ("-1 10 279 1542 36 3\n", Bag(iid=-1, capacity=279, min_cost=1542, items=[])),
        ("-1 10 279 36 3\n", Bag(iid=-1, capacity=279, min_cost=float("inf"), items=[])),
    ]
)
def test_bag_load(rawline, expected):
    bag = Bag.from_line(rawline)
    assert bag.iid == expected.iid
    assert bag.min_cost == expected.min_cost
    assert bag.capacity == expected.capacity


def load_bag_data(x_file, y_file):
    x_path = os.path.join(THIS_DIR, x_file)
    y_path = os.path.join(THIS_DIR, y_file)

    with open(x_path) as x, open(y_path) as y:
        return tuple(zip(x.readlines(), y.readlines()))


@pytest.mark.parametrize(
    "bag_def, bag_sol",
    load_bag_data("data/NR/NR4_inst.dat", "data/NR/NK4_sol.dat"),
    # [
    #     (
    #         '-1 4 46 324 36 3 43 1129 202 94 149 2084\n',
    #         '1 4 1129 0 1 0 0 \n'
    #     ),
    # ]
)
def test_bag_solve(bag_def, bag_sol):
    bag = Bag.from_line(bag_def)
    bag.solve()

    parsed = [int(v) for v in bag_sol.strip().split(" ")]
    iid, count, target_cost, *target_items = parsed
    assert bag.iid == -1 * iid
    assert tuple(int(i) for i in bag.best_solution) == tuple(i for i in target_items)
    assert bag.best_cost == target_cost
