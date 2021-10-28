from dataclasses import dataclass


@dataclass
class Item:
    weight: int
    cost: int
    index: int

    @property
    def cw_ratio(self) -> float:
        return self.cost / self.weight
