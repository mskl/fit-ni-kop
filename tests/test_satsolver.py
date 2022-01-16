import numpy as np
import pytest
from satsolver.instance import Instance
from satsolver.genetic import GeneticSolver
from satsolver.utils import load_solutions


def test_load_solutions():
    solutions = load_solutions("data/wuf-N1/wuf20-78-N-opt.dat")
    assert solutions["uf20-0305"]


@pytest.mark.parametrize("InstanceCls", [Instance, GeneticSolver])
def test_load_instance(InstanceCls):
    inst = InstanceCls.from_file("data/wuf-N1/wuf20-78-N1/wuf20-0305.mwcnf")
    assert inst.probid == "uf20-78/uf20-0305"
    assert inst.solution_id == "uf20-0305"


def test_solves():
    inst = Instance(
        probid="uf20-78/uf20-0305",
        weights=np.array([1, 2, 3]),
        clauses=np.array([[2, 1, 3], [-1, 2, 3], [-1, -2, 3]])
    )

    assert inst.solves(np.array([0, 0, 1]))
    assert not inst.solves(np.array([0, 0, 0]))
    assert inst.optscore(np.array([1, 1, 1])) == 6


def test_solves_example():
    inst = Instance.from_file("data/wuf-N1/wuf20-78-N1/wuf20-0305.mwcnf")
    sols = load_solutions("data/wuf-N1/wuf20-78-N-opt.dat")

    solution = sols[inst.solution_id]
    assert inst.solves(solution.solution)
    assert inst.optscore(solution.solution) == solution.optscore

