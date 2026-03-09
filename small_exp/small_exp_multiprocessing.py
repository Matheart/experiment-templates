"""
Multiprocessing execution of hyperparameter sweep.
Runs 90 experiments (3 lr × 5 batch × 6 width) in parallel processes with detailed timing.
width and batch size make program run with different time
"""

import subprocess
import time
import multiprocessing as mp
from itertools import product
import os
import argparse


def run_experiment(args):
    """Run single experiment and return timing info."""
    lr, wd, batch, width, idx, total = args
    log_file = f"logs/exp_lr{lr}_bs{batch}_w{width}.log"
    
    expt_start = time.time()
    cmd = ["uv", "run", "python", "small_exp.py",
           "--lr", str(lr),
           "--batch-size", str(batch),
           "--wd", str(wd),
           "--width", str(width)]
    
    with open(log_file, 'wb') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    
    expt_time = time.time() - expt_start
    
    print(f"[{idx}/{total}] lr={lr}, batch={batch}, width={width} | "
          f"Time: {expt_time:.1f}s")
    
    return expt_time


def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter sweep with multiprocessing')
    parser.add_argument('--workers', type=int, default=20, help='Number of parallel workers (default: 20)')
    args = parser.parse_args()
    
    # Hyperparameter ranges
    lr_values = [1e-3, 1e-4, 1e-5]
    wd = 1e-5  # Fixed weight decay
    batch_sizes = [32, 64, 128, 256, 512]
    widths = [64, 128, 256, 512, 1024, 2048]
    
    combinations = [(lr, wd, batch, width) 
                    for batch, width, lr in product(batch_sizes, widths, lr_values)]
    #                for lr, batch, width in product(lr_values, batch_sizes, widths)]
    total = len(combinations)
    
    # Create tasks with indices for tracking
    tasks = [(lr, wd, batch, width, idx, total) 
             for idx, (lr, wd, batch, width) in enumerate(combinations, 1)]
    
    print(f"Multiprocessing execution: {total} experiments")
    print(f"Workers: {args.workers} parallel processes")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    with mp.Pool(processes=args.workers) as pool:
        expt_times = pool.map(run_experiment, tasks)
    
    total_time = time.time() - start_time
    
    print(f"\n=== Multiprocessing Results ===")
    print(f"Total experiments: {total}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Average time per experiment: {sum(expt_times)/len(expt_times):.2f} seconds")
    print(f"Min experiment time: {min(expt_times):.2f} seconds")
    print(f"Max experiment time: {max(expt_times):.2f} seconds")


if __name__ == '__main__':
    main()
