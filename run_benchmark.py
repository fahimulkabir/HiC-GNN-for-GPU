import time
import subprocess
import pandas as pd
import os
import re
import sys

# ======================
# CONFIGURATION
# ======================
DATASET = "data/GM12878_1mb_chr19_list.txt" 
RUNS = 3
RESULT_FILE = "benchmark_results.csv"

def parse_output(output_str):
    """
    Scans the script output for the 'Final Result' line.
    Handles standard floats (0.94) and scientific notation (1.2e-5).
    """
    dSCC = None
    alpha = None
    
    # IMPROVED REGEX: Handles scientific notation (e.g., 1.23e-04) just in case
    # Looks for "Best dSCC:", optional whitespace, then captures number
    match = re.search(r"Best dSCC:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", output_str)
    if match:
        dSCC = float(match.group(1))
        
    match_alpha = re.search(r"Alpha:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", output_str)
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
    
    process = subprocess.run(
        [sys.executable, "-m", "src.main", DATASET],
        capture_output=True,
        text=True
    )
    
    end_time = time.time()
    elapsed = end_time - start_time

    if process.returncode != 0:
        print("FAILED")
        print("Error Log:\n", process.stderr)
        continue

    dSCC, alpha = parse_output(process.stdout)

    # ADDED: Safety check for parsing errors
    if dSCC is None:
        print("Warning: Script finished but couldn't parse dSCC score.")
        print("Output snippet:", process.stdout[-200:]) # Print last 200 chars for debugging
    else:
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
    
    if os.path.exists(RESULT_FILE):
        # Append mode: header=False prevents writing columns again
        df.to_csv(RESULT_FILE, mode='a', header=False, index=False)
    else:
        # Write mode: standard write with header
        df.to_csv(RESULT_FILE, index=False)

    print("\nBenchmark Summary:")
    print(df)
    print(f"\nSaved to {RESULT_FILE}")
else:
    print("\nNo successful runs recorded.")