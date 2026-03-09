"""
Ablation study for learning rate, weight decay, and seeds.
Runs experiments with different combinations of lr, wd, and random seeds.
"""

import subprocess
import time
import multiprocessing as mp
from itertools import product
import argparse
import os


def run_experiment(args):
    """Run single experiment and return timing info."""
    lr, wd, seed, idx, total = args
    log_file = f"logs/ablation_lr{lr}_wd{wd}_seed{seed}.log"
    
    expt_start = time.time()
    cmd = ["uv", "run", "python", "small_exp.py",
           "--lr", str(lr),
           "--batch-size", "32",  # Fixed batch size
           "--wd", str(wd),
           "--width", "64",  # Fixed width
           "--seed", str(seed)]  # Add seed argument
    
    with open(log_file, 'wb') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    
    expt_time = time.time() - expt_start
    
    print(f"[{idx}/{total}] lr={lr}, wd={wd}, seed={seed} | "
          f"Time: {expt_time:.1f}s")
    
    return expt_time


def main():
    parser = argparse.ArgumentParser(description='Run ablation study with multiprocessing')
    parser.add_argument('--workers', type=int, default=20, help='Number of parallel workers (default: 20)')
    args = parser.parse_args()
    
    # Ablation ranges
    lr_values = [1e-3, 1e-4, 1e-5, 1e-6]  # 4 learning rates
    wd_values = [0.0, 1e-6, 1e-5]  # 3 weight decays (including 0)
    seeds = [42, 123, 456]  # 3 random seeds
    
    combinations = [(lr, wd, seed) 
                    for lr, wd, seed in product(lr_values, wd_values, seeds)]
    total = len(combinations)
    
    # Create tasks with indices for tracking
    tasks = [(lr, wd, seed, idx, total) 
             for idx, (lr, wd, seed) in enumerate(combinations, 1)]
    
    print(f"Ablation study: {total} experiments")
    print(f"Learning rates: {lr_values}")
    print(f"Weight decays: {wd_values}")
    print(f"Seeds: {seeds}")
    print(f"Workers: {args.workers} parallel processes")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    with mp.Pool(processes=args.workers) as pool:
        expt_times = pool.map(run_experiment, tasks)
    
    total_time = time.time() - start_time
    
    print(f"\n=== Ablation Results ===")
    print(f"Total experiments: {total}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Average time per experiment: {sum(expt_times)/len(expt_times):.2f} seconds")
    print(f"Min experiment time: {min(expt_times):.2f} seconds")
    print(f"Max experiment time: {max(expt_times):.2f} seconds")


if __name__ == '__main__':
    main()
