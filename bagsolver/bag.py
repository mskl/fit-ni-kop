from typing import List, Union, Optional
import numpy as np

from .item import Item


class Bag:
    def __init__(self, iid: int, capacity: int, min_cost: Union[int, float], items: Optional[List[Item]]):
        self.iid = iid
        self.capacity = capacity
        self.min_cost = min_cost
        self.items = items
        self.size = len(self.items)

        # kind of bag problem
        self.best_solution = np.zeros(self.size)
        self.optimizations = None
        self.best_cost = None
        self.proposal = None
        self.opcount = None

    @classmethod
    def from_line(cls, line: str) -> "Bag":
        parsed = [int(v) for v in line.strip().split(" ")]

        min_cost = float("inf")
        if len(parsed) % 2 == 0:
            iid, count, capacity, min_cost, *items = parsed
        else:
            iid, count, capacity, *items = parsed

        parsed_items = [Item(weight, cost) for weight, cost in zip(items[0::2], items[1::2])]
        return cls(iid, capacity, min_cost, parsed_items)

    def evaluate(self, proposal: List[bool]) -> (int, int):
        selection = [i for (i, p) in zip(self.items, proposal) if p]
        cost = sum(i.cost for i in selection)
        weight = sum(i.weight for i in selection)
        return cost, weight

    def solve_bb(self, optimizations=None) -> int:
        """Solve using branch&bound approach. If optimizations are None, a brute-force will be used."""
        self.best_cost = 0
        self.opcount = 0
        self.proposal = np.zeros(self.size)
        self.optimizations = optimizations or set()
        self._solve_bb(0, 0, 0)
        return self.best_cost

    def _solve_bb(self, index: int, tweight: int, tcost: int) -> None:
        self.opcount = self.opcount + 1

        weight = tweight + self.items[index-1].weight * self.proposal[index-1]
        cost = tcost + self.items[index-1].cost * self.proposal[index-1]

        weight_passed = weight <= self.capacity

        if "weight" in self.optimizations and not weight_passed:
            return

        if weight_passed and cost > self.best_cost:
            self.best_cost = cost
            self.best_solution = np.zeros(self.size)
            self.best_solution[:index + 1] = self.proposal[:index + 1]

        if "residuals" in self.optimizations:
            if residual_items := self.items[index:]:
                residual_cost = sum(i.cost for i in residual_items)
                if (cost + residual_cost) < self.best_cost:
                    return

        # End of recursion
        if index >= self.size:
            return

        self.proposal[index] = True
        self._solve_bb(index + 1, weight, cost)
        self.proposal[index] = False
        self._solve_bb(index + 1, weight, cost)
