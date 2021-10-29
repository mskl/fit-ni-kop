import numpy as np
import time
import os
import random
import pandas as pd
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

from bagsolver.bag import Bag


def solve_line(line, optimizations, size, itype):
    bag = Bag.from_line(line)
    start_time = time.perf_counter()
    bag.solve_branch_bound(optimizations=optimizations, strict=True)
    elapsed = time.perf_counter() - start_time
    return bag.opcount, elapsed, optimizations, size, itype


sizes = [4, 10, 15, 20, 22, 25] # 27, 30, 32, 35, 37

subsample = 250

if __name__ == '__main__':
    results = []
    tasks = []

    for size in sizes:
        for itype in ["ZR", "NR"]:
            filepath = f"./data/{itype}/{itype}{size}_inst.dat"
            with open(filepath) as x:
                problems = x.readlines()

            # shuffle consistently
            random.Random(4).shuffle(problems)

            for optimizations in [set(), {"residuals", "weight"}]:
                for line in problems[:subsample]:
                    tasks.append((line, optimizations, size, itype))

    random.shuffle(tasks)

    futures = []
    with tqdm(total=len(tasks)) as pbar:
        with ProcessPoolExecutor(max_workers=5) as executor:
            for task in tasks:
                futures.append(executor.submit(solve_line, *task))
                
            for future in as_completed(futures):
                results.append(future.result())
                pbar.update(1)


    df = pd.DataFrame(results, columns=["opcount", "elapsed", "optimizations", "size", "itype"])
    df.to_csv("results.csv", index=False)