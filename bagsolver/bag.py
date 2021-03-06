from functools import lru_cache
import numpy as np
import math
import copy

from .base import BagBase


class Bag(BagBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.best_cost = None
        self.optimizations = None
        self.proposal = None
        self.best_solution = None

        self.initialize()

    def initialize(self) -> None:
        self.best_cost = 0
        self.optimizations = set()
        self.proposal = np.zeros(self.size)
        self.best_solution = np.zeros(self.size)

    def solve_ftapas(self, epsilon: float):
        maxcost = max(self.items, key=lambda x: x.cost).cost
        k = (epsilon * maxcost) / self.size
        items = copy.deepcopy(self.items)

        for item in self.items:
            item.cost = math.ceil(item.cost / k)

        self.solve_dynamic_weight()

        # Recover the saved values on items
        self.items = items

        return self.evaluate(self.best_solution)[0]

    def solve_dynamic_weight(self) -> int:
        total_cost = sum([_.cost for _ in self.items])
        lookup_table = np.full((self.size, total_cost+1), math.inf)

        lookup_table[0, 0] = 0
        if self.items[0].cost != 0:
            lookup_table[0, self.items[0].cost] = self.items[0].weight

        total_rows = lookup_table.shape[1]

        for col, item in enumerate(self.items):
            if col == 0:
                continue

            for row in range(total_rows):
                lookup_table[col, row] = lookup_table[col-1, row]
                if item.cost <= row:
                    if lookup_table[col-1, row - item.cost] != math.inf:
                        prev = lookup_table[col-1, row - item.cost] + item.weight
                        if prev < lookup_table[col, row]:
                            lookup_table[col, row] = prev

        for cost in reversed(range(total_rows)):
            weight = lookup_table[self.size-1, cost]
            if weight <= self.capacity and cost > self.best_cost:
                self.best_cost = cost

        # Reconstruct back the best solution from table
        residual = self.best_cost
        for i in reversed(range(self.size)):
            if i != 0:
                if lookup_table[i-1, residual] != lookup_table[i, residual]:
                    residual -= self.items[i].cost
                    self.best_solution[i] = 1
            else:
                if residual != 0:
                    self.best_solution[0] = 1

        value, weight = self.evaluate(self.best_solution)

        return value

    def solve_dynamic_cost(self) -> int:
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

    def solve_bruteforce(self) -> int:
        self._solve_bb(0, 0, 0)
        return self.best_cost

    def solve_branch_bound(self) -> int:
        """Solve using branch&bound approach. If optimizations are None, a brute-force will be used."""
        self.optimizations = {"weight", "residuals"}
        return self.solve_bruteforce()

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
            residual_items = self.items[index:]
            if residual_items:
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
