import numpy as np
import pytest
from bagsolver.bag import Bag
from bagsolver.utils import load_bag_data


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


def get_dataset():
    return load_bag_data("data/NK/NK10_inst.dat", "data/NK/NK10_sol.dat")[:10]


@pytest.mark.parametrize("bag_def, bag_sol", get_dataset())
@pytest.mark.parametrize("optimizations", [None, {"residuals"}, {"weight"}, {"weight", "residuals"}])
def test_bag_solve(bag_def, bag_sol, optimizations):
    bag = Bag.from_line(bag_def)
    res = bag.solve_bb(optimizations=optimizations)

    parsed = [int(v) for v in bag_sol.strip().split(" ")]
    iid, count, target_cost, *target_items = parsed
    assert bag.iid == iid
    assert tuple(int(i) for i in bag.best_solution) == tuple(i for i in target_items)
    assert res == target_cost


@pytest.mark.parametrize("bag_def, bag_sol", get_dataset())
@pytest.mark.parametrize("redux", [False, True])
def test_bag_solve_greedy(bag_def, bag_sol, redux):
    bag = Bag.from_line(bag_def)
    res = bag.solve_greedy(redux)

    parsed = [int(v) for v in bag_sol.strip().split(" ")]
    iid, count, target_cost, *target_items = parsed
    assert bag.iid == iid


@pytest.mark.parametrize("bag_def, bag_sol", get_dataset())
def test_bag_dynamic_cost(bag_def, bag_sol):
    bag = Bag.from_line(bag_def)
    res = bag.solve_dynamic_cost()

    parsed = [int(v) for v in bag_sol.strip().split(" ")]
    iid, count, target_cost, *target_items = parsed
    assert res == target_cost
    assert bag.iid == iid


@pytest.mark.parametrize("bag_def, bag_sol", get_dataset())
def test_bag_dynamic_weight(bag_def, bag_sol):
    bag = Bag.from_line(bag_def)
    res = bag.solve_dynamic_weight()

    parsed = [int(v) for v in bag_sol.strip().split(" ")]
    iid, count, target_cost, *target_items = parsed
    assert res == target_cost
    assert bag.iid == iid


@pytest.mark.parametrize("bag_def, bag_sol", get_dataset())
def test_bag_ftapas(bag_def, bag_sol):
    bag = Bag.from_line(bag_def)
    res = bag.solve_ftapas(.3)

    parsed = [int(v) for v in bag_sol.strip().split(" ")]
    iid, count, target_cost, *target_items = parsed
    assert res == target_cost
    assert bag.iid == iid
