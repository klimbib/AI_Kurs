"""Whole pipeline: ingest -> search -> rag.

Usage:
    python main.py                         # ingest, evaluate search, answer all questions in eval/testset.json
                                           # (answers are written to output/rag/answers.json), then evals.py
                                           # (eval/evals.json, baseline gate) and evals.py --calibrate
                                           # (eval/calibration_evals.json)
    python main.py "what component are available?"     # ingest, then one question: answer with sources in the terminal
    python main.py --groups purchasing "what is the cost of the component?"
    python main.py --full                  # ingest everything again, ignoring the manifest
"""
import argparse
import sys

import evals
import ingest
import rag
import search


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--groups", default="all")
    ap.add_argument("--full", action="store_true", help="ignore the manifest and re-ingest every file")
    args = ap.parse_args()

    print("=== 1. ingest ===")
    ingest.print_report(ingest.run(args.full))

    if args.query:
        print("\n=== 2. search (all configurations) ===")
        search.main([args.query, "--groups", args.groups])
        print(f"\n=== 3. rag (search method: {rag.METHOD}) ===")
        rag.main(["--groups", args.groups, args.query])      # then a dialogue (last 3 turns)
    else:
        # testset.json: groups come from the file, no dialogue
        print("\n=== 2. search ===")
        search.main([])
        print(f"\n=== 3. rag (search method: {rag.METHOD}) ===")
        rag.main([])
        gate_failed = False
        print("\n=== 4. evals (comparison with baseline, gate) ===")
        try:
            evals.main([])
        except SystemExit as e:             # a red gate exits with 1: still run the calibration
            gate_failed = bool(e.code)
        print("\n=== 5. evals --calibrate ===")
        evals.main(["--calibrate"])
        if gate_failed:
            sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        sys.exit(f"Not implemented yet: {e}")
