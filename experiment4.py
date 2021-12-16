from typing import Callable
from bagsolver.genetic import GeneticBagSolver
from concurrent.futures import as_completed, ProcessPoolExecutor
from tqdm import tqdm
import time
import pandas as pd

from bagsolver.utils import load_bag_data


def solve_instance(bagdef, bagsol, batch_size=30, mutation_rate=0.04, runtime=3):
    bag = GeneticBagSolver.from_line(bagdef, batch_size=batch_size, mutation_rate=mutation_rate)
    pool = bag.new_pool_random()

    start = time.time()

    while time.time() - start < runtime:
        # Called just to save the scores
        pool = bag.genetic_iteration(pool)
        batch_stats = bag.pool_fitness(pool)

    best_score = bag.best_score
    target = int(bagsol.split(" ")[2])

    return target, best_score, batch_size, mutation_rate


def run(workers: int = 8, executor_cls: Callable = ProcessPoolExecutor) -> pd.DataFrame:
    tasks, records = [], []
    data = load_bag_data("data/NK/NK40_inst.dat", "data/NK/NK40_sol.dat")
    for batch_size in [10, 20, 40, 80, 160, 320, 640, 1280]:
        for mutation_rate in [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.30]:
            for bagdef, bagsol in data:
                tasks.append((bagdef, bagsol, batch_size, mutation_rate))

    with tqdm(total=len(tasks)) as pbar:
        for i in range(0, len(tasks), 2000):
            futures = []
            chunk = tasks[i:i + 2000]
            with executor_cls(max_workers=workers) as executor:
                for task in chunk:
                    futures.append(executor.submit(solve_instance, *task))

                for future in as_completed(futures):
                    records.append(future.result())
                    pbar.update(1)

    return pd.DataFrame(records, columns=["target", "best_score", "batch_size", "mutation_rate"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Solve experiment 3.")
    parser.add_argument("-w", "--workers", type=int, default=7)
    parser.add_argument("-n", "--name", type=str, default="results4.csv")

    args = parser.parse_args()

    df = run(args.workers, ProcessPoolExecutor)
    print(f"Saving results into {args.name}.")
    df.to_csv(args.name, index=False)
