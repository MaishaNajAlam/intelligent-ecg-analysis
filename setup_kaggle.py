"""
Kaggle / Google Colab setup runner for the ECG project.

HOW TO USE ON KAGGLE
─────────────────────
1. Create a new Kaggle Notebook (Python, GPU T4 x2 enabled).
2. In the right panel → "Add Data" → search "PTB-XL" →
   add "PTB-XL: A Large Publicly Available ECG Dataset".
3. Upload this entire project folder as a Kaggle Dataset OR
   use "Add Data" → paste GitHub URL (if your repo is public).
4. In the first notebook cell, run:
       !python setup_kaggle.py
   This file will handle all path wiring, pip installs, and
   run the Step 1 checkpoint automatically.

HOW TO USE ON GOOGLE COLAB
───────────────────────────
1. Open a new Colab notebook with GPU runtime
   (Runtime → Change runtime type → T4 GPU).
2. Mount Drive:
       from google.colab import drive
       drive.mount('/content/drive')
3. Clone or upload the project:
       !git clone <your-github-repo-url> /content/ecg-project
       %cd /content/ecg-project
4. Run:
       !python setup_kaggle.py --colab --ptbxl_dir /path/to/ptbxl
"""

import os
import sys
import subprocess
import argparse


# ── Detect platform ──────────────────────────────────────────────────
ON_KAGGLE = os.path.exists("/kaggle/input")
ON_COLAB  = "google.colab" in sys.modules or os.path.exists("/content")


def find_kaggle_ptbxl():
    """Return the PTB-XL path from the Kaggle input directory."""
    kaggle_input = "/kaggle/input"
    for entry in os.listdir(kaggle_input):
        # Match any dataset whose folder name contains 'ptb' and 'xl'
        lower = entry.lower()
        if "ptb" in lower and ("xl" in lower or "ecg" in lower):
            candidate = os.path.join(kaggle_input, entry)
            # Verify it has the expected files
            if os.path.exists(os.path.join(candidate, "ptbxl_database.csv")):
                return candidate
    return None


def pip_install():
    print("=" * 60)
    print("Installing requirements...")
    print("=" * 60)
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "-r", "requirements.txt", "-q"])
    print("✔  All packages installed.\n")


def set_ptbxl_env(ptbxl_dir: str):
    os.environ["PTBXL_DIR"] = ptbxl_dir
    print(f"✔  PTBXL_DIR set to: {ptbxl_dir}\n")


def run_step1_checkpoint():
    print("=" * 60)
    print("Running Step 1 checkpoint: data exploration")
    print("=" * 60)
    subprocess.check_call([sys.executable, "-m", "src.data", "--explore"])
    print("\n✔  Step 1 checkpoint PASSED.")
    print("   Check outputs/example_ecg_plot.png for the 12-lead plot.\n")


def run_step2_checkpoint():
    print("=" * 60)
    print("Running Step 2 checkpoint: encoder test")
    print("=" * 60)
    subprocess.check_call([sys.executable, "-m", "src.encoder", "--test"])
    print("\n✔  Step 2 checkpoint PASSED.\n")


def main():
    parser = argparse.ArgumentParser(description="ECG project environment setup")
    parser.add_argument("--colab",      action="store_true", help="Running on Colab")
    parser.add_argument("--ptbxl_dir",  default=None,
                        help="Manual path to PTB-XL directory (Colab / local use)")
    parser.add_argument("--skip_step1", action="store_true",
                        help="Skip the Step 1 data checkpoint")
    parser.add_argument("--skip_step2", action="store_true",
                        help="Skip the Step 2 encoder checkpoint")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  ECG Project — Environment Setup")
    print("=" * 60)
    print(f"  Platform : {'Kaggle' if ON_KAGGLE else 'Colab' if ON_COLAB else 'Local'}")
    print(f"  Python   : {sys.version.split()[0]}")
    print("=" * 60 + "\n")

    # ── 1. Install packages ──────────────────────────────────────────
    pip_install()

    # ── 2. Locate PTB-XL ────────────────────────────────────────────
    if args.ptbxl_dir:
        ptbxl_dir = args.ptbxl_dir
    elif ON_KAGGLE:
        ptbxl_dir = find_kaggle_ptbxl()
        if ptbxl_dir is None:
            print("❌  Could not auto-detect PTB-XL in /kaggle/input/.")
            print("   Make sure you added the PTB-XL dataset to this notebook.")
            print("   Then re-run with: python setup_kaggle.py --ptbxl_dir /kaggle/input/<folder-name>")
            sys.exit(1)
        print(f"✔  Auto-detected PTB-XL at: {ptbxl_dir}")
    else:
        # Default local path
        ptbxl_dir = os.path.join(os.path.dirname(__file__), "data", "ptbxl")
        print(f"ℹ  Using default local PTB-XL path: {ptbxl_dir}")
        print("   (Override with --ptbxl_dir if it's elsewhere)\n")

    set_ptbxl_env(ptbxl_dir)

    # ── 3. Step 1 checkpoint ─────────────────────────────────────────
    if not args.skip_step1:
        run_step1_checkpoint()
    else:
        print("⏭  Skipping Step 1 checkpoint (--skip_step1 set).\n")

    # ── 4. Step 2 checkpoint ─────────────────────────────────────────
    if not args.skip_step2:
        run_step2_checkpoint()
    else:
        print("⏭  Skipping Step 2 checkpoint (--skip_step2 set).\n")

    print("=" * 60)
    print("  Setup complete! Next steps:")
    print("=" * 60)
    print("  Train classifier (Step 3):")
    print("    python -m src.classifier --train")
    print("    python -m src.classifier --evaluate")
    print()
    print("  Train report generator (Step 4):")
    print("    python -m src.report_generator --train")
    print("    python -m src.report_generator --evaluate")
    print()
    print("  Or run everything at once:")
    print("    python run_pipeline.py --all")
    print()
    print("  Launch the web app (Step 5) after training:")
    print("    python -m src.app")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
