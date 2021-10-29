from typing import List, Union, Optional
import numpy as np
import math

from .item import Item

from functools import lru_cache


class Bag:
    def __init__(self, iid: int, capacity: int, min_cost: Union[int, float], items: Optional[List[Item]]):
        self.iid = iid
        self.capacity = capacity
        self.min_cost = min_cost
        self.items = items
        self.size = len(self.items)

        self.best_solution = np.zeros(self.size)
        self.optimizations = None
        self.best_cost = None
        self.proposal = None

    def reset(self) -> None:
        self.best_cost = 0
        self.optimizations = None
        self.proposal = np.zeros(self.size)
        self.best_solution = np.zeros(self.size)

    @classmethod
    def from_line(cls, line: str) -> "Bag":
        parsed = [int(v) for v in line.strip().split(" ")]

        min_cost = float("inf")
        if len(parsed) % 2 == 0:
            iid, count, capacity, min_cost, *items = parsed
        else:
            iid, count, capacity, *items = parsed

        parsed_items = [
            Item(weight, cost, index)
            for index, (weight, cost)
            in enumerate(zip(items[0::2], items[1::2]))
        ]
        return cls(iid, capacity, min_cost, parsed_items)

    def evaluate(self, proposal: List[bool]) -> (int, int):
        selection = [i for (i, p) in zip(self.items, proposal) if p]
        cost = sum(i.cost for i in selection)
        weight = sum(i.weight for i in selection)
        return cost, weight

    def solve_ftapas(self, epsilon: float):
        maxcost = max(self.items, key=lambda x: x.cost).cost
        k = (epsilon * maxcost) / self.size

        for item in self.items:
            item.k = k

        res = self.solve_dynamic_weight(k)

        for item in self.items:
            item.k = 1.0

        return res

    def solve_dynamic_weight(self, k: float = 1.0) -> int:
        self.reset()

        # self.total_cost = sum([_.cost for _ in self.items])
        # self.table = np.full(())

        for cost in range(self.total_cost, 1, -1):
            minweight = self._solve_dynamic_weight(self.size - 1, math.floor(cost / k))
            if cost >= self.best_cost and minweight <= self.capacity:
                self.best_cost = cost
        return self.best_cost

    @lru_cache(maxsize=None)
    def _solve_dynamic_weight(self, index: int, cost: int) -> int:
        if index < 0:
            return math.inf if cost > 0 else 0
        item = self.items[index]
        s0 = self._solve_dynamic_weight(index - 1, cost)
        s1 = self._solve_dynamic_weight(index - 1, cost - item.cost) + item.weight
        return min(s0, s1)

    def solve_dynamic_cost(self) -> int:
        self.reset()
        return self._solve_dynamic_cost(self.size - 1, self.capacity)

    @lru_cache(maxsize=None)
    def _solve_dynamic_cost(self, index: int, capacity: int) -> int:
        if index < 0 or capacity <= 0:
            return 0
        item = self.items[index]
        if item.weight > capacity:
            return self._solve_dynamic_cost(index-1, capacity)
        item_no = self._solve_dynamic_cost(index-1, capacity)
        item_yes = item.cost + self._solve_dynamic_cost(index-1, capacity - item.weight)
        return max(item_no, item_yes)

    def solve_greedy(self, redux=False) -> int:
        used_capacity = 0
        items_cost = 0
        for item in sorted(self.items, key=lambda x: x.cw_ratio, reverse=True):
            if used_capacity + item.weight <= self.capacity:
                self.best_solution[item.index] = 1
                used_capacity += item.weight
                items_cost += item.cost

        if redux:
            # Select only most valuable item that fits the bag
            filtered = [_ for _ in self.items if _.weight <= self.capacity]
            if not filtered:
                return items_cost

            bestitem = max(filtered, key=lambda x: x.cost)
            if items_cost < bestitem.cost:
                self.best_solution = np.zeros_like(self.best_solution)
                self.best_solution[bestitem.index] = 1
                return bestitem.cost

        return items_cost

    def solve_bb(self, optimizations=None) -> int:
        """Solve using branch&bound approach. If optimizations are None, a brute-force will be used."""
        self.optimizations = optimizations or set()
        self._solve_bb(0, 0, 0)
        return self.best_cost

    def _solve_bb(self, index: int, tweight: int, tcost: int) -> None:
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
