import numpy as np
from typing import Tuple


class Instance:
    def __init__(self, probid: str, weights: np.ndarray, clauses: np.ndarray):
        self.probid = probid
        self.weights = weights

        # Plenty of calculations just to prepare clauses into nice format
        sorted_indices = np.argsort(np.abs(clauses), axis=1)
        sorted_clauses = np.take_along_axis(clauses, sorted_indices, axis=1)
        self.clauses_signs = np.sign(sorted_clauses)
        # Get rid of indexing from 1 :rolleyes:
        abs_clauses = np.abs(sorted_clauses)
        self.clauses_centered = abs_clauses - 1

    @staticmethod
    def _params_from_file(filepath: str) -> Tuple[str, np.ndarray, np.ndarray]:
        probid, weights, clauses = None, None, []
        with open(filepath) as fp:
            for line in fp.readlines():
                line = line.strip().rstrip("0 ")
                if line.startswith("c SAT instance "):
                    probid = line[len("c SAT instance "): -len(".cnf")]
                elif line.startswith("c"):
                    continue
                elif line.startswith("p"):
                    continue  # probname = line.lstrip("p ")
                elif line.startswith("w"):
                    weights = np.array([int(_) for _ in line.lstrip("w ").split(" ")])
                else:
                    clauses.append(np.array([int(_) for _ in line.split(" ")]))
        return probid, weights, np.array(clauses)

    @classmethod
    def from_file(cls, filepath: str) -> "Instance":
        """Parse the formula on given filepath. Uses newlines (not zeros) as delimiters."""
        return Instance(*cls._params_from_file(filepath))

    @property
    def solution_id(self) -> str:
        return self.probid.split("/")[1]

    def optscore(self, proposal) -> int:
        return np.where(proposal < 0, 0, proposal) @ self.weights

    def _clausules_correct(self, proposal: np.ndarray) -> np.ndarray:
        # Proposal is automatically broadcasted to match dims
        extracted = np.take(proposal, self.clauses_centered)
        return self.clauses_signs == extracted

    def solves(self, proposal: np.ndarray) -> bool:
        return np.all(self._clausules_correct(proposal).max(axis=1))
