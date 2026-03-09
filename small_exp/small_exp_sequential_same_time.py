"""
Sequential ablation study for learning rate, weight decay, and seeds.
Runs experiments one at a time with detailed timing.
"""

import subprocess
import time
from itertools import product


def main():
    # Ablation ranges
    lr_values = [1e-3, 1e-4, 1e-5, 1e-6]  # 4 learning rates
    wd_values = [0.0, 1e-6, 1e-5]  # 3 weight decays (including 0)
    seeds = [42, 123, 456]  # 3 random seeds
    
    combinations = [(lr, wd, seed) 
                    for lr, wd, seed in product(lr_values, wd_values, seeds)]
    total = len(combinations)
    
    print(f"Sequential ablation study: {total} experiments")
    print(f"Learning rates: {lr_values}")
    print(f"Weight decays: {wd_values}")
    print(f"Seeds: {seeds}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    expt_times = []
    
    for idx, (lr, wd, seed) in enumerate(combinations, 1):
        log_file = f"logs/ablation_lr{lr}_wd{wd}_seed{seed}.log"
        expt_start = time.time()
        
        cmd = ["uv", "run", "python", "small_exp.py",
               "--lr", str(lr),
               "--batch-size", "32",  # Fixed batch size
               "--wd", str(wd),
               "--width", "64",  # Fixed width
               "--seed", str(seed)]
        
        with open(log_file, 'wb') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        
        expt_time = time.time() - expt_start
        expt_times.append(expt_time)
        
        elapsed = time.time() - start_time
        eta = (elapsed / idx) * (total - idx)
        avg_time = sum(expt_times) / len(expt_times)
        
        print(f"[{idx}/{total}] lr={lr}, wd={wd}, seed={seed} | "
              f"Time: {expt_time:.1f}s, Avg: {avg_time:.1f}s, ETA: {eta/60:.1f}min")
    
    total_time = time.time() - start_time
    print(f"\n=== Sequential Ablation Results ===")
    print(f"Total experiments: {total}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Average time per experiment: {sum(expt_times)/len(expt_times):.2f} seconds")
    print(f"Min experiment time: {min(expt_times):.2f} seconds")
    print(f"Max experiment time: {max(expt_times):.2f} seconds")


if __name__ == '__main__':
    main()
