"""Single-edit check of the generated reports.

Every retained distorted report of the error types omission, direction, size, typo and unit is compared word by
word with its source report. A report is listed for exclusion when
  - nothing was changed ("no change"), or
  - two or more separate places were changed ("multiple edits").
The comparison ignores letter case, and changes separated by at most one unchanged word count as one place.
Insertion is not checked because its prompt also removes statements that contradict the inserted finding.

Usage:
  python single_edit_check.py --csv_dir <folder with dspy_nodule.csv and dspy_effusion.csv> --out exclusion_list.csv
The input files are the outputs of error_generation_dspy.py.
"""
import argparse
import difflib
import re

import pandas as pd

tok = lambda s: re.findall(r"\w+|[^\w\s]", str(s).lower())
norm = lambda s: re.sub(r"\s+", " ", str(s)).strip().lower()


def hunks(a, b):
    """Changed regions of a relative to b; regions separated by at most one unchanged token are merged."""
    hs = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if hs and i1 - hs[-1][1] <= 1:
            hs[-1] = (hs[-1][0], i2)
        else:
            hs.append((i1, i2))
    return hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    groups = {k: pd.read_csv(f"{args.csv_dir}/dspy_{k}.csv", low_memory=False) for k in ["nodule", "effusion"]}
    spec = [("nodule", "omission", "omission_nodule"), ("effusion", "omission", "omission_effusion"),
            ("nodule", "direction", "direction_nodule"), ("effusion", "direction", "direction_effusion")]
    spec += [(g, t, t) for g in ["nodule", "effusion"] for t in ["size", "typo", "unit"]]

    rows = []
    for grp, error_type, prefix in spec:
        df = groups[grp]
        df = df[pd.to_numeric(df[prefix + "_classification"], errors="coerce") == 1]   # retained reports only
        for _, r in df.iterrows():
            source, distorted = str(r["lung_section"]), str(r[prefix + "_distorted_report"])
            volume = str(r["id"])
            if norm(source) == norm(distorted):
                rows.append((volume, error_type, "no change"))
            elif len(hunks(tok(source), tok(distorted))) >= 2:
                rows.append((volume, error_type, "multiple edits"))
    ex = pd.DataFrame(rows, columns=["volume", "error_type", "reason"])
    assert not ex.duplicated(["volume", "error_type"]).any()
    ex.to_csv(args.out, index=False)
    print(ex.groupby(["error_type", "reason"]).size().unstack(fill_value=0))
    print("total", len(ex))


if __name__ == "__main__":
    main()
