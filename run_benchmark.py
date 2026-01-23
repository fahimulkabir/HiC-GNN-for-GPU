import time
import subprocess
import pandas as pd
import os
import re
import sys

# ======================
# CONFIGURATION
# ======================
# Ensure this matches your actual file path in data/
DATASET = "data/GM12878_1mb_chr19_list.txt" 
RUNS = 3
RESULT_FILE = "benchmark_results.csv"

def parse_output(output_str):
    """
    Scans the script output for the 'Final Result' line.
    Example: 'Final Result -> Best dSCC: 0.9464 (Alpha: 0.4)'
    """
    dSCC = None
    alpha = None
    
    # Regex to find the Best dSCC score
    match = re.search(r"Best dSCC:\s*([0-9\.]+)", output_str)
    if match:
        dSCC = float(match.group(1))
        
    # Regex to find the Alpha
    match_alpha = re.search(r"Alpha:\s*([0-9\.]+)", output_str)
    if match_alpha:
        alpha = float(match_alpha.group(1))

    return dSCC, alpha

# ======================
# MAIN LOOP
# ======================
results = []

print(f"   Starting Benchmark on: {DATASET}")
print(f"   Runs: {RUNS}\n")

for i in range(RUNS):
    print(f"Run {i+1}/{RUNS}...", end=" ", flush=True)

    start_time = time.time()
    
    # COMMAND: python -m src.main data/...
    # capture_output=True lets us grab print statements to parse dSCC
    process = subprocess.run(
        [sys.executable, "-m", "src.main", DATASET],
        capture_output=True,
        text=True
    )
    
    end_time = time.time()
    elapsed = end_time - start_time

    # Check if it crashed
    if process.returncode != 0:
        print("FAILED")
        print("Error Log:\n", process.stderr)
        continue

    # Parse results from the output text
    dSCC, alpha = parse_output(process.stdout)

    print(f"Done in {elapsed:.1f}s | dSCC: {dSCC} (Alpha: {alpha})")

    results.append({
        "run": i + 1,
        "time_sec": elapsed,
        "dSCC": dSCC,
        "best_alpha": alpha,
        "device": "cuda" if "cuda" in process.stdout.lower() else "cpu"
    })

# ======================
# SAVE & SUMMARY
# ======================
if results:
    df = pd.DataFrame(results)
    
    # Append to existing file if it exists, else create new
    if os.path.exists(RESULT_FILE):
        df.to_csv(RESULT_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(RESULT_FILE, index=False)

    print("\nBenchmark Summary:")
    print(df)
    print(f"\nSaved to {RESULT_FILE}")
else:
    print("\nNo successful runs recorded.")