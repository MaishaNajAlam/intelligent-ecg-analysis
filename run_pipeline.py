"""
Runs the whole pipeline end-to-end, in order, matching the project plan's
6 build steps. Useful once PTB-XL is downloaded and you just want to go.

Usage:
    python run_pipeline.py --all
    python run_pipeline.py --step 1 2 3
"""
import argparse
import subprocess
import sys

STEPS = {
    1: [sys.executable, "-m", "src.data", "--explore"],
    2: [sys.executable, "-m", "src.encoder", "--test"],
    3: [sys.executable, "-m", "src.classifier", "--train"],
    3.5: [sys.executable, "-m", "src.classifier", "--evaluate"],
    4: [sys.executable, "-m", "src.report_generator", "--train"],
    4.5: [sys.executable, "-m", "src.report_generator", "--evaluate"],
    6: [sys.executable, "-m", "src.evaluate_full"],
}

DESCRIPTIONS = {
    1: "Step 1: Environment and data setup (checkpoint)",
    2: "Step 2: Load encoder, test feature extraction (checkpoint)",
    3: "Step 3: Train classifier head",
    3.5: "Step 3: Evaluate classifier on test set",
    4: "Step 4: Train report generator (adapter + BART)",
    4.5: "Step 4: Evaluate report generator (BLEU/ROUGE)",
    6: "Step 6: Systematic evaluation and error analysis",
}


def run_step(step):
    print(f"\n{'='*70}\n{DESCRIPTIONS[step]}\n{'='*70}")
    subprocess.run(STEPS[step], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run every step in order")
    parser.add_argument("--step", nargs="+", type=float, help="Run specific step(s), e.g. --step 1 2 3")
    args = parser.parse_args()

    if args.all:
        for s in [1, 2, 3, 3.5, 4, 4.5, 6]:
            run_step(s)
        print("\nAll steps complete. Run `python -m src.app` to launch the web interface (Step 5).")
    elif args.step:
        for s in args.step:
            run_step(s)
    else:
        print("Nothing to do. Use --all or --step <numbers>. See README.md for the full walkthrough.")
