from typing import Callable
from bagsolver.bag import Bag
from bagsolver.utils import load_bag_data, parse_solution
from concurrent.futures import as_completed, ProcessPoolExecutor
from tqdm import tqdm
import time
import pandas as pd
import random

RUNS = {
    "branch_bound": ("solve_branch_bound", {}),
    "greedy_simple": ("solve_greedy", {"redux": False}),
    "greedy_redux": ("solve_greedy", {"redux": True}),
    "dynamic_cost": ("solve_dynamic_cost", {}),
    "dynamic_weight": ("solve_dynamic_weight", {}),
    "solve_ftapas_03": ("solve_ftapas", {"epsilon": 0.3}),
    "solve_ftapas_05": ("solve_ftapas", {"epsilon": 0.5}),
    "solve_ftapas_07": ("solve_ftapas", {"epsilon": 0.7}),
}


def solve_line(bagdef, bagsol, dataset, size, name, params, key):
    bag = Bag.from_line(bagdef)
    iid, count, target_cost, target_items = parse_solution(bagsol)

    start = time.time()
    bag.initialize()
    res = getattr(bag, name)(**params)
    elapsed = time.time() - start
    delta = abs(res - target_cost)

    return dataset, size, key, elapsed, delta


def run(runs: dict, workers: int = 5, executor_class: Callable=ProcessPoolExecutor, subsample: int = None) -> pd.DataFrame:
    tasks, futures, records = [], [], []

    for dataset in ["NK", "ZKC", "ZKW"]:
        for size in [4, 10, 15, 20, 22, 25, 27, 30]:
            data = load_bag_data(
                f"data/{dataset}/{dataset}{size}_inst.dat",
                f"data/{dataset}/{dataset}{size}_sol.dat"
            )
            select_data = data[:subsample] if subsample else data
            for bagdef, bagsol in select_data:
                for key, (name, params) in runs.items():
                    tasks.append((bagdef, bagsol, dataset, size, name, params, key))

    random.shuffle(tasks)
    with tqdm(total=len(tasks)) as pbar:
        with executor_class(max_workers=workers) as executor:
            for task in tasks:
                futures.append(executor.submit(solve_line, *task))

            for future in as_completed(futures):
                records.append(future.result())
                pbar.update(1)

    return pd.DataFrame(records, columns=["dataset", "size", "key", "elapsed", "delta"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Solve experiment 2.")
    parser.add_argument("-w", "--workers", type=int, default=7)
    parser.add_argument("-s", "--subsample", type=int, default=None)
    parser.add_argument("-n", "--name", type=str, default="results2.csv")

    args = parser.parse_args()

    df = run(RUNS, args.workers, ProcessPoolExecutor, args.subsample)
    print(f"Saving results into {args.name}.")
    df.to_csv(args.name, index=False)

