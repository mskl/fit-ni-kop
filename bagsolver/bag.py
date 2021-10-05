from typing import List, Union, Optional
import numpy as np

from .exceptions import StrictSolutionFound
from .item import Item


class Bag:
    def __init__(self, iid: int, capacity: int, min_cost: Union[int, float], items: Optional[List[Item]]):
        self.iid = iid
        self.capacity = capacity
        self.min_cost = min_cost
        self.items = items

        # kind of bag problem
        self.best_solution = np.zeros(self.size)
        self.optimizations = None
        self.best_cost = None
        self.opcount = None
        self.strict = None

    @property
    def size(self):
        return len(self.items)

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

    def solve(self, optimizations=None, strict=False) -> bool:
        self.best_cost = 0
        self.opcount = 0
        self.optimizations = optimizations or set()
        try:
            self._solve(index=0, proposal=np.zeros(self.size), strict=strict)
        except StrictSolutionFound:
            return True

        if self.best_cost > self.min_cost:
            return True
        return False

    def _solve(self, index: int, proposal: np.ndarray, strict: bool) -> None:
        self.opcount = self.opcount + 1

        cost, weight = self.evaluate(proposal[:index+1])
        if (weight <= self.capacity) and (cost > self.best_cost):
            self.best_cost = cost
            self.best_solution = np.zeros(self.size)
            self.best_solution[:index + 1] = proposal[:index + 1]

        # End of recursion
        if index >= self.size:
            return

        # Bag is already too heavy too heavy
        if "weight" in self.optimizations and weight > self.capacity:
            return

        if "residuals" in self.optimizations:
            if residual_items := self.items[index:]:
                residual_cost = sum(i.cost for i in residual_items)
                if cost + residual_cost < self.best_cost:
                    return

        # Strict mode would do
        if self.strict and cost >= self.min_cost:
            raise StrictSolutionFound

        proposal[index] = True
        self._solve(index+1, proposal, strict)
        proposal[index] = False
        self._solve(index+1, proposal, strict)
