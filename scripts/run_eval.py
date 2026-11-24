"""Run all evaluations. Usage: python scripts/run_eval.py [--limit N] [--skip-ragas]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation import chunking_compare, ragas_eval

p = argparse.ArgumentParser()
p.add_argument("--limit", type=int, default=None, help="only the first N test questions")
p.add_argument("--skip-ragas", action="store_true", help="skip the LLM-judged RAGAS run")
args = p.parse_args()

print("=== chunking comparison ===")
chunking_compare.main()
if not args.skip_ragas:
    print("\n=== RAGAS: hybrid vs vector-only ===")
    ragas_eval.main(args.limit)
