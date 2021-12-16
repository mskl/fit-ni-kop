from typing import Callable
from bagsolver.genetic import GeneticBagSolver
from concurrent.futures import as_completed, ProcessPoolExecutor
from tqdm import tqdm
import time
import pandas as pd

from bagsolver.utils import load_bag_data, parse_solution


def solve_instance(bagdef, bagsol, batch_size=30, mutation_rate=0.04, init_method="random", runtime=3):
    bag = GeneticBagSolver.from_line(bagdef, batch_size=batch_size, mutation_rate=mutation_rate)
    if init_method == "random":
        pool = bag.new_pool_random()
    elif init_method == "naive":
        pool = bag.new_pool_naive()
    else:
        raise ValueError("Unknown init method", init_method)

    start = time.time()

    while time.time() - start < runtime:
        pool = bag.genetic_iteration(pool)
        # Called just to save the scores
        batch_stats = bag.pool_fitness(pool)

    runtime = time.time() - start

    iid, count, target, target_items = parse_solution(bagsol)
    return target, bag.best_score, batch_size, mutation_rate, init_method, runtime


def run(workers: int = 8, subsample: int = 50, runtime: int = 3, executor_cls: Callable = ProcessPoolExecutor) -> pd.DataFrame:
    tasks, records = [], []
    data = load_bag_data("data/NK/NK40_inst.dat", "data/NK/NK40_sol.dat")[:subsample]
    for init_method in ["random", "naive"]:
        for batch_size in [5, 10, 20, 40, 80, 160, 320, 640, 1280]:
            for mutation_rate in [0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32]:
                for bagdef, bagsol in data:
                    tasks.append((bagdef, bagsol, batch_size, mutation_rate, init_method, runtime))

    CHUNKING = 1000 # Helps to clear memory between runs
    with tqdm(total=len(tasks)) as pbar:
        for i in range(0, len(tasks), CHUNKING):
            futures = []
            chunk = tasks[i:i + CHUNKING]
            with executor_cls(max_workers=workers) as executor:
                for task in chunk:
                    futures.append(executor.submit(solve_instance, *task))

                for future in as_completed(futures):
                    records.append(future.result())
                    pbar.update(1)

    return pd.DataFrame(records, columns=["target", "best_score", "batch_size", "mutation_rate", "init_method", "runtime"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Solve experiment 3.")
    parser.add_argument("-w", "--workers", type=int, default=7)
    parser.add_argument("-s", "--subsample", type=int, default=50)
    parser.add_argument("-r", "--runtime", type=int, default=3)
    parser.add_argument("-n", "--name", type=str, default="results4.csv")

    args = parser.parse_args()

    df = run(args.workers, args.subsample, args.runtime, ProcessPoolExecutor)
    print(f"Saving results into {args.name}.")
    df.to_csv(args.name, index=False)
