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


@pytest.mark.parametrize("bag_def, bag_sol", load_bag_data("data/NR/NR4_inst.dat", "data/NR/NK4_sol.dat")[:50])
@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("optimizations", [None, {"weight"}, {"weight", "residuals"}])
def test_bag_solve(bag_def, bag_sol, strict, optimizations):
    bag = Bag.from_line(bag_def)
    res = bag.solve(optimizations=optimizations, strict=strict)

    parsed = [int(v) for v in bag_sol.strip().split(" ")]
    iid, count, target_cost, *target_items = parsed
    assert bag.iid == -1 * iid
    if not strict:
        assert tuple(int(i) for i in bag.best_solution) == tuple(i for i in target_items)
        assert bag.best_cost == target_cost
    else:
        should_pass = bag.min_cost < target_cost
        assert res == should_pass
