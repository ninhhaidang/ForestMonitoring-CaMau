#!/usr/bin/env python3
"""
Simple progress monitor for SNUNet training
Run: python monitor_progress.py
"""

import os
import re
import time
from pathlib import Path

MODEL_NAME = 'SNUNet-CD'
WORK_DIR = 'experiments/snunet'
MAX_ITERS = 6400
REFRESH_INTERVAL = 5  # seconds

def find_log():
    """Find latest log file"""
    log_files = list(Path(WORK_DIR).rglob('*.log'))
    if not log_files:
        return None
    return str(sorted(log_files, key=lambda x: x.stat().st_mtime)[-1])

def parse_iterations(log_path):
    """Count completed iterations"""
    pattern = r'Iter\(train\)\s+\[\s*(\d+)/(\d+)\]'
    iterations = []

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    iterations.append(int(match.group(1)))
    except:
        pass

    return iterations[-1] if iterations else 0

def get_eta(log_path):
    """Get ETA from log"""
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in reversed(lines):
            match = re.search(r'eta:\s+([\d:]+)', line)
            if match:
                return match.group(1)
    except:
        pass
    return "--:--:--"

def show_progress(current, total, eta="--:--:--"):
    """Display progress bar"""
    os.system('cls' if os.name == 'nt' else 'clear')

    if current == 0:
        print("\n" + "="*70)
        print("  Dang khoi tao training...")
        print("  Doi batch dau tien (co the mat 2-5 phut)")
        print("="*70 + "\n")
        return

    percent = (current / total) * 100
    bar_length = 50
    filled = int(bar_length * current / total)
    bar = '#' * filled + '-' * (bar_length - filled)

    print("\n" + "="*70)
    print(f"  {MODEL_NAME} TRAINING")
    print("="*70)
    print()
    print(f"  [{bar}] {percent:.1f}%")
    print()
    print(f"  Iterations: {current:,} / {total:,}")
    print(f"  ETA: {eta}")
    print()
    print(f"  Last update: {time.strftime('%H:%M:%S')}")
    print("="*70)
    print("\n  Press Ctrl+C to stop monitoring")

def main():
    print(f"\nAuto-refresh every {REFRESH_INTERVAL}s")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            log_path = find_log()

            if not log_path:
                os.system('cls' if os.name == 'nt' else 'clear')
                print("\n  Waiting for log file...")
                time.sleep(REFRESH_INTERVAL)
                continue

            current_iter = parse_iterations(log_path)
            eta = get_eta(log_path)

            show_progress(current_iter, MAX_ITERS, eta)

            if current_iter >= MAX_ITERS:
                print("\n" + "="*70)
                print("  TRAINING COMPLETE!")
                print("="*70 + "\n")
                break

            time.sleep(REFRESH_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n  Monitoring stopped")
        print("  Training still running in background!\n")

if __name__ == '__main__':
    main()
