import random
from typing import List

from bagsolver.base import BagBase
import numpy as np

from bagsolver.item import Item


class GeneticBagSolver(BagBase):
	def __init__(self, *args, batch_size: int, mutation_rate: float, **kwargs):
		super().__init__(*args, **kwargs)

		self.batch_size = batch_size
		self.mutation_rate = mutation_rate

		self.w = np.array([i.weight for i in self.items])
		self.v = np.array([i.cost for i in self.items])

		self.best_score = 0
		self.best_instance = np.zeros_like(self.v)

	@classmethod
	def from_line(cls, line: str, batch_size=100, mutation_rate=0.04) -> "GeneticBagSolver":
		return cls(*cls.params_from_line(line), batch_size=batch_size, mutation_rate=mutation_rate)

	def value(self, p: np.ndarray) -> float:
		return self.v@p

	def weight(self, p: np.ndarray) -> float:
		return self.w@p

	def valid(self, p: np.ndarray) -> bool:
		return self.weight(p) <= self.capacity

	def fitness_naive(self, p: np.ndarray) -> float:
		"""Asymmetric implementation of fitness function."""
		if not self.valid(p):
			return 0
		value = self.value(p)
		if value > self.best_score:
			self.best_score = value
			self.best_instance = p.copy()
		return value

	def new_instance_naive(self) -> np.ndarray:
		return np.ones_like(self.v)

	def new_instance_random(self) -> np.ndarray:
		"""Create a random binary vector that passes the capacity."""
		proposal = np.zeros_like(self.v)
		total_w = 0
		for i in np.random.permutation(len(self.v)):
			if total_w + self.w[i] > self.capacity:
				break
			total_w += self.w[i]
			proposal[i] = 1
		assert self.valid(proposal)
		return proposal

	@staticmethod
	def crossover_mask(p: np.ndarray) -> np.ndarray:
		"""Generate crossover mask. Always include at least one 1 or 0."""
		mask = np.zeros_like(p, dtype="int8")
		point = np.random.randint(len(mask) - 1)
		mask[:point + 1] = 1
		return mask

	@staticmethod
	def mask_inv(m: np.ndarray) -> np.ndarray:
		"""Invert the given mask and return a copy."""
		return m.copy() * -1 + 1

	def mutate(self, p: np.ndarray) -> np.array:
		"""Mutate the p array with the preset probability."""
		mutation_mask = (np.random.rand(len(p)) < self.mutation_rate).astype("int8")
		return np.where(mutation_mask == 0, p, self.mask_inv(p))

	# noinspection PyTypeChecker
	def best_candidate(self, candidates: List[np.ndarray]) -> np.ndarray:
		return max(candidates, key=self.fitness_naive)

	def recombine(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
		"""Recombine using one point crossover and mutate. Return best child or fallback to parent."""
		mask_org = self.crossover_mask(p1)
		mask_inv = self.mask_inv(mask_org)
		c1 = self.mutate(mask_org*p1 + mask_inv*p2)
		c2 = self.mutate(mask_inv*p1 + mask_org*p2)
		if self.valid(c1) or self.valid(c2):
			return self.best_candidate([c1, c2])
		return self.best_candidate([p1, p2])

	def genetic_iteration(self, parents: List[np.ndarray]) -> List[np.ndarray]:
		"""Use roulette wheel selection to generate the next batch of instances."""
		total_fitness = sum([self.fitness_naive(p) for p in parents])
		cumm_fitness = np.cumsum([self.fitness_naive(p) for p in parents])

		children = []
		while len(children) < len(parents):
			p_idx1 = np.searchsorted(cumm_fitness, np.random.randint(total_fitness))
			p_idx2 = np.searchsorted(cumm_fitness, np.random.randint(total_fitness))
			# NOTE: Parents can be same, but it's unlikely
			children.append(self.recombine(parents[p_idx1], parents[p_idx2]))

		random.shuffle(children)
		return children

	def new_pool_random(self) -> List[np.ndarray]:
		"""Generate initial pool of random parents."""
		return [self.new_instance_random() for i in range(self.batch_size)]

	def new_pool_naive(self) -> List[np.ndarray]:
		"""Generate initial pool of naive parents."""
		return [self.new_instance_naive() for i in range(self.batch_size)]

	def pool_fitness(self, pool: List[np.ndarray]) -> (float, float, float):
		"""Return min, max, mean fitness of the pool"""
		pool_fitness = [self.fitness_naive(i) for i in pool]
		return np.min(pool_fitness), np.max(pool_fitness), np.mean(pool_fitness)
