import time
import subprocess
import pandas as pd
import os
import re

# ======================
# CONFIGURATION
# ======================
DATASET = "Data/GM12878_1mb_chr19_list.txt"
SCRIPT = "HiC-GNN_main.py"
OUTPUT_LOG = "Outputs/GM12878_1mb_chr19_list_log.txt"

RUNS = 3
MODE = "cpu"  # change to "gpu" when testing GPU

RESULT_FILE = "benchmark_results.csv"


# ======================
# Helper: parse metrics
# ======================
def parse_log(log_path):
    dSCC, loss = None, None

    if not os.path.exists(log_path):
        return None, None

    with open(log_path, "r") as f:
        for line in f:
            if "dSCC" in line:
                match = re.search(r"dSCC:\s*([0-9\.]+)", line)
                if match:
                    dSCC = float(match.group(1))
            if "loss" in line.lower():
                match = re.search(r"loss:\s*([0-9\.]+)", line.lower())
                if match:
                    loss = float(match.group(1))

    return dSCC, loss


# ======================
# Run benchmark
# ======================
results = []

for i in range(RUNS):
    print(f"\nRun {i+1}/{RUNS} ({MODE.upper()})")

    start = time.time()

    env = os.environ.copy()

    if MODE == "cpu":
        env["USE_GPU"] = "0"
    else:
        env["USE_GPU"] = "1"

    subprocess.run(["python", SCRIPT, DATASET], env=env)

    elapsed = time.time() - start

    dSCC, loss = parse_log(OUTPUT_LOG)

    results.append({
        "run": i + 1,
        "mode": MODE,
        "time_sec": elapsed,
        "dSCC": dSCC,
        "loss": loss
    })


# ======================
# Save results
# ======================
df = pd.DataFrame(results)

if os.path.exists(RESULT_FILE):
    old = pd.read_csv(RESULT_FILE)
    df = pd.concat([old, df], ignore_index=True)

df.to_csv(RESULT_FILE, index=False)

print("\nBenchmark results:")
print(df)
print(f"\nSaved to {RESULT_FILE}")
