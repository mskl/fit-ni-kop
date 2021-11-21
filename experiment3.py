from typing import Callable
from bagsolver.bag import Bag
from concurrent.futures import as_completed, ProcessPoolExecutor
from tqdm import tqdm
import time
import pandas as pd
import random


def solve_line(instance, method, args):
    bag = Bag.from_line(instance)

    start = time.time()
    bag.initialize()
    result = getattr(bag, method)()
    elapsed = time.time() - start

    return instance, method, result, elapsed, args


def run(tasks_csv, workers: int = 5, executor_class: Callable = ProcessPoolExecutor) -> pd.DataFrame:
    tasks, records = [], []

    tasks_df = pd.read_csv(tasks_csv)
    for _, row in tasks_df.iterrows():
        tasks.append([row["instance"], row["method"], row["args"]])

    random.shuffle(tasks)

    with tqdm(total=len(tasks)) as pbar:
        for i in range(0, len(tasks), 2000):
            futures = []
            chunk = tasks[i:i + 2000]
            with executor_class(max_workers=workers) as executor:
                for task in chunk:
                    futures.append(executor.submit(solve_line, *task))

                for future in as_completed(futures):
                    records.append(future.result())
                    pbar.update(1)

    return pd.DataFrame(records, columns=["instance", "method", "result", "elapsed", "args"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Solve experiment 3.")
    parser.add_argument("-t", "--tasks", type=str, required=True)
    parser.add_argument("-w", "--workers", type=int, default=7)
    parser.add_argument("-n", "--name", type=str, default="results3.csv")

    args = parser.parse_args()

    df = run(args.tasks, args.workers, ProcessPoolExecutor)
    print(f"Saving results into {args.name}.")
    df.to_csv(args.name, index=False)
