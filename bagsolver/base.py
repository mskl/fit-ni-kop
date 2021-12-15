import random
from typing import List, Union, Iterable

from .item import Item


class BagBase:
	def __init__(self, iid: int, capacity: int, min_cost: Union[int, float], items: List[Item]):
		self.iid = iid
		self.capacity = capacity
		self.min_cost = min_cost
		self.items = items
		self.size = len(self.items)

	def shuffle(self) -> None:
		random.shuffle(self.items)

	def evaluate(self, proposal: Iterable[bool]) -> (int, int):
		selection = [i for (i, p) in zip(self.items, proposal) if p]
		cost = sum(i.cost for i in selection)
		weight = sum(i.weight for i in selection)
		return cost, weight

	@classmethod
	def params_from_line(cls, line: str) -> tuple:
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

		return iid, capacity, min_cost, parsed_items

	@classmethod
	def from_line(cls, line: str) -> "BagBase":
		return cls(*cls.params_from_line(line))
