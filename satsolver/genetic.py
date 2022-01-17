import glob
import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

from satsolver.instance import Instance


class GeneticSolver(Instance):
	def __init__(
		self,
		*args,
		batch_size: int,
		mutation_rate: float,
		init_type: str,
		fitness_type: str,
		**kwargs,
	):
		super().__init__(*args, **kwargs)

		self.batch_size = batch_size
		self.mutation_rate = mutation_rate
		self.init_type = init_type
		self.fitness_type = fitness_type

		self.best_score = 0
		self.best_instance = None

	@classmethod
	def from_file(
		cls,
		filepath: str,
		batch_size: int = 100,
		mutation_rate: float = .04,
		init_type: str = "uniform",
		fitness_type: str = "sum_or_nothing"
	) -> "GeneticSolver":
		probid, weights, clauses = cls._params_from_file(filepath=filepath)
		return cls(
			probid=probid,
			weights=weights,
			clauses=clauses,
			batch_size=batch_size,
			mutation_rate=mutation_rate,
			init_type=init_type,
			fitness_type=fitness_type
		)

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

	def best_candidate(self, candidates: List[np.ndarray]) -> np.ndarray:
		random.shuffle(candidates)
		return max(candidates, key=self.fitness)

	def recombine(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
		"""Do one point crossover and mutate. Return best child or fallback to parent."""
		mask_org = self.crossover_mask(p1)
		mask_inv = self.mask_inv(mask_org)
		c1 = self.mutate(mask_org*p1 + mask_inv*p2)
		c2 = self.mutate(mask_inv*p1 + mask_org*p2)
		if self.solves(c1) or self.solves(c2):
			return self.best_candidate([c1, c2])
		return self.best_candidate([p1, p2, c1, c2])

	def genetic_iteration(self, parents: List[np.ndarray]) -> List[np.ndarray]:
		"""Use roulette wheel selection to generate the next batch of instances."""
		total_fitness = sum([self.fitness(p) for p in parents])
		cumm_fitness = np.cumsum([self.fitness(p) for p in parents])

		children = []
		while len(children) < len(parents):
			if total_fitness == 0:
				p_idx1 = np.random.randint(len(parents))
				p_idx2 = np.random.randint(len(parents))
			else:
				p_idx1 = np.searchsorted(cumm_fitness, np.random.randint(total_fitness))
				p_idx2 = np.searchsorted(cumm_fitness, np.random.randint(total_fitness))
			# NOTE: Parents can be same, but it's unlikely
			children.append(self.recombine(parents[p_idx1], parents[p_idx2]))

		random.shuffle(children)
		return children

	def new_instance(self) -> np.ndarray:
		if self.init_type == "allfalse":
			return np.ones_like(self.weights) * -1
		elif self.init_type == "uniform":
			return np.random.randint(2, size=len(self.weights)) * 2 - 1
		else:
			raise ValueError(f"Unknown init method: {self.init_type}")

	def fitness(self, p: np.ndarray) -> float:
		solves = self.solves(p)
		if self.fitness_type == "sum_or_nothing":
			value = self.optscore(p) if solves else 0
		elif self.fitness_type == "correct_count":
			correct_count = np.sum(self._clausules_correct(p))
			correct_weighted = np.sum(np.take(self.weights, self.clauses_centered) * self._clausules_correct(p))
			value = correct_count if not solves else correct_count + correct_weighted
		else:
			raise ValueError(f"Unknown fitness function {self.fitness_type}")

		if value > self.best_score:
			self.best_score = value
			self.best_instance = p.copy()
		return value

	def new_pool(self) -> List[np.ndarray]:
		"""Generate initial pool of naive parents."""
		return [self.new_instance() for _ in range(self.batch_size)]

	def pool_fitness(self, pool: List[np.ndarray]) -> List[float]:
		"""Return values of fitness for all elements in pool"""
		return [self.fitness(i) for i in pool]

	@staticmethod
	def pool_stats(pf: List[float]) -> Tuple[float, float, float, float]:
		"""Returns stats in form of min, max, mean, median"""
		# noinspection PyTypeChecker
		return np.min(pf), np.max(pf), np.mean(pf), np.median(pf)

	def solved(self) -> bool:
		if self.best_instance is not None:
			return self.solves(self.best_instance)
		return False

