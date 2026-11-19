"""Released question-answer files.

Reads the items written by vqa_construction.ipynb, removes the distorted reports listed by single_edit_check.py
and writes one flat record per item:
    id, volume, patient, task, error_type, group, question, report, answer

Usage:
  python build_release.py --item_dir <folder with error_full_list_final_new_level{1,2,3}.json> \
      --csv_dir <folder with dspy_nodule.csv, dspy_effusion.csv, dspy_none.csv> \
      --exclusion exclusion_list.csv --out_dir ../data
"""
import argparse
import json
import os
import re

import pandas as pd

SEP = "\nHere is the medical report: '''"
TASKS = [("classification", "1"), ("detection", "2"), ("correction", "3")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item_dir", required=True)
    ap.add_argument("--csv_dir", required=True)
    ap.add_argument("--exclusion", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # group of each scan
    group = {}
    for name, label in [("nodule", "nodule"), ("effusion", "effusion"), ("none", "control")]:
        for volume in pd.read_csv(f"{args.csv_dir}/dspy_{name}.csv", low_memory=False, usecols=["id"])["id"]:
            assert volume not in group, f"{volume} belongs to two groups"
            group[volume] = label

    ex = pd.read_csv(args.exclusion)
    excluded = set(zip(ex["volume"], ex["error_type"]))

    for task, level in TASKS:
        items = json.load(open(f"{args.item_dir}/error_full_list_final_new_level{level}.json"))
        out, n_excluded = [], 0
        for x in items:
            question_turn, answer_turn = x["conversations"]
            assert question_turn["level"] == level, x["id"]
            if (x["image"], question_turn["type"]) in excluded:
                n_excluded += 1
                continue
            value = question_turn["value"]
            assert value.count(SEP) == 1 and value.endswith("'''"), x["id"]
            question, report = value.split(SEP)
            out.append({"id": f"{task}_{len(out) + 1:05d}",
                        "volume": x["image"],
                        "patient": re.match(r"(train_\d+)_", x["image"]).group(1),
                        "task": task,
                        "error_type": "none" if question_turn["type"] == "normal" else question_turn["type"],
                        "group": group[x["image"]],
                        "question": question,
                        "report": report[:-3],
                        "answer": answer_turn["value"]})
        json.dump(out, open(f"{args.out_dir}/{task}.json", "w"), indent=1, ensure_ascii=False)
        print(f"{task}: {len(items)} items, {n_excluded} removed, {len(out)} written")


if __name__ == "__main__":
    main()
