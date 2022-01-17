import random
from concurrent.futures import as_completed, ProcessPoolExecutor
from tqdm import tqdm
import time
import pandas as pd
import glob

from satsolver.genetic import GeneticSolver


def solve_instance(batch_size, mutation_rate, problem_path, init_type, fitness_type):
    solver = GeneticSolver.from_file(
        filepath=problem_path,
        batch_size=batch_size,
        mutation_rate=mutation_rate,
        fitness_type=fitness_type,
        init_type=init_type
    )

    pool = solver.new_pool()
    solved_time = None
    start = time.time()
    while (time.time() - start) < 3:
        pool = solver.genetic_iteration(pool)
        if solver.solved():
            solved_time = time.time() - start
            break
    return batch_size, mutation_rate, problem_path, init_type, fitness_type, solved_time


def run(n_workers: int = 32) -> pd.DataFrame:
    selected = glob.glob("data/wuf-N1/wuf20-78-N1/*")[:128]
    tasks, records = [], []
    CHUNKING = 3000

    tasks = []
    for batch_size in [5, 10, 20, 40, 80, 160, 320, 640, 1280]:
        for mutation_rate in [0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32]:
            for init_type in ["allfalse", "uniform"]:
                for fitness_type in ["sum_or_nothing", "correct_count"]:
                    for problem_path in selected:
                        tasks.append(
                            {
                                "batch_size": batch_size,
                                "mutation_rate": mutation_rate,
                                "problem_path": problem_path,
                                "init_type": init_type,
                                "fitness_type": fitness_type
                            }
                        )
    random.shuffle(tasks)
    # Helps to clear memory between runs
    with tqdm(total=len(tasks)) as pbar:
        for i in range(0, len(tasks), CHUNKING):
            futures = []
            chunk = tasks[i:i + CHUNKING]
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                for task in chunk:
                    futures.append(executor.submit(solve_instance, **task))

                for future in as_completed(futures):
                    records.append(future.result())
                    pbar.update(1)

    return pd.DataFrame(
        records,
        columns="batch_size,mutation_rate,problem_path,init_type,fitness_type,solved_time".split(",")
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Solve experiment 5.")
    parser.add_argument("-w", "--workers", type=int, default=7)

    args = parser.parse_args()

    df = run(args.workers)
    dfname = "pilot5v3.csv"
    print(f"Saving results into {dfname}.")
    df.to_csv(dfname, index=False)
