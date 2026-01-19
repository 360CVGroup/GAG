#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import Counter


def main():
    ap = argparse.ArgumentParser(description="Count records per dataset_name in a JSONL file.")
    ap.add_argument("jsonl_path", help="Path to input .jsonl file")
    ap.add_argument("--encoding", default="utf-8", help="File encoding (default: utf-8)")
    ap.add_argument("--csv-out", default=None, help="Optional output CSV path")
    args = ap.parse_args()

    counter = Counter()
    total = 0
    bad_json = 0
    missing_field = 0

    with open(args.jsonl_path, "r", encoding=args.encoding) as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue

            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                bad_json += 1
                continue

            total += 1
            name = obj.get("dataset_name")
            if name is None:
                missing_field += 1
                name = "__MISSING__"
            counter[name] += 1

    print(f"Total parsed records: {total}")
    if bad_json:
        print(f"JSON parse failed lines: {bad_json}")
    if missing_field:
        print(f"Missing dataset_name records: {missing_field}")

    print("\nCounts by dataset_name:")
    for name, cnt in counter.most_common():
        print(f"  {name}\t{cnt}")

    if args.csv_out:
        import csv
        with open(args.csv_out, "w", encoding="utf-8", newline="") as wf:
            w = csv.writer(wf)
            w.writerow(["dataset_name", "count"])
            for name, cnt in counter.most_common():
                w.writerow([name, cnt])
        print(f"\nSaved CSV to: {args.csv_out}")


if __name__ == "__main__":
    main()
