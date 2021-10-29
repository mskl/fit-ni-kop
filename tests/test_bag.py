import pytest
from bagsolver.bag import Bag
from bagsolver.utils import load_bag_data, parse_solution


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


def get_dataset(subsample=100):
    return load_bag_data("data/NK/NK10_inst.dat", "data/NK/NK10_sol.dat")[:subsample]


@pytest.mark.parametrize("bag_def, bag_sol", get_dataset())
@pytest.mark.parametrize("optimizations", [None, {"residuals"}, {"weight"}, {"weight", "residuals"}])
def test_bag_solve(bag_def, bag_sol, optimizations):
    res = Bag.from_line(bag_def).solve_branch_bound(optimizations=optimizations)

    iid, count, target_cost, target_items = parse_solution(bag_sol)
    assert res == target_cost


@pytest.mark.parametrize("bag_def, bag_sol", get_dataset())
@pytest.mark.parametrize("redux", [False, True])
def test_bag_solve_greedy(bag_def, bag_sol, redux):
    res = Bag.from_line(bag_def).solve_greedy(redux)

    iid, count, target_cost, target_items = parse_solution(bag_sol)
    assert (res - target_cost) < 1000


@pytest.mark.parametrize("bag_def, bag_sol", get_dataset())
def test_bag_dynamic_cost(bag_def, bag_sol):
    res = Bag.from_line(bag_def).solve_dynamic_cost()

    iid, count, target_cost, target_items = parse_solution(bag_sol)
    assert res == target_cost


@pytest.mark.parametrize("bag_def, bag_sol", get_dataset())
def test_bag_dynamic_weight(bag_def, bag_sol):
    res = Bag.from_line(bag_def).solve_dynamic_weight()

    iid, count, target_cost, target_items = parse_solution(bag_sol)
    assert res == target_cost


@pytest.mark.parametrize("bag_def, bag_sol", get_dataset())
def test_bag_ftapas(bag_def, bag_sol):
    res = Bag.from_line(bag_def).solve_ftapas(.3)

    iid, count, target_cost, target_items = parse_solution(bag_sol)
    assert (res - target_cost) < 100
