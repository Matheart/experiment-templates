"""
Sequential execution of hyperparameter sweep.
Runs 90 experiments (3 lr × 5 batch × 6 width) one at a time with detailed timing.
"""

import subprocess
import time
from itertools import product


def main():
    # Hyperparameter ranges
    lr_values = [1e-3, 1e-4, 1e-5]
    wd = 1e-5  # Fixed weight decay
    batch_sizes = [32, 64, 128, 256, 512]
    widths = [64, 128, 256, 512, 1024, 2048]
    
    combinations = [(lr, wd, batch, width) for lr, batch, width in product(lr_values, batch_sizes, widths)]
    total = len(combinations)
    
    print(f"Sequential execution: {total} experiments")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    expt_times = []
    
    for idx, (lr, wd, batch, width) in enumerate(combinations, 1):
        log_file = f"logs/exp_lr{lr}_bs{batch}_w{width}.log"
        expt_start = time.time()
        
        cmd = ["uv", "run", "python", "small_exp.py",
               "--lr", str(lr),
               "--batch-size", str(batch),
               "--wd", str(wd),
               "--width", str(width)]
        
        with open(log_file, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        
        expt_time = time.time() - expt_start
        expt_times.append(expt_time)
        
        elapsed = time.time() - start_time
        eta = (elapsed / idx) * (total - idx)
        avg_time = sum(expt_times) / len(expt_times)
        
        print(f"[{idx}/{total}] lr={lr}, batch={batch}, width={width} | "
              f"Time: {expt_time:.1f}s, Avg: {avg_time:.1f}s, ETA: {eta/60:.1f}min")
    
    total_time = time.time() - start_time
    print(f"\n=== Sequential Results ===")
    print(f"Total experiments: {total}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Average time per experiment: {sum(expt_times)/len(expt_times):.2f} seconds")
    print(f"Min experiment time: {min(expt_times):.2f} seconds")
    print(f"Max experiment time: {max(expt_times):.2f} seconds")


if __name__ == '__main__':
    main()
