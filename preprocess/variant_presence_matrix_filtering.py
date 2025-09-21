#!/usr/bin/env python3
# filter_raw_matrix.py
import pandas as pd
import sys, pathlib, re

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {pathlib.Path(sys.argv[0]).name} <variant_presence_matrix.csv>", file=sys.stderr)
        sys.exit(1)

    in_path = pathlib.Path(sys.argv[1]).resolve()
    out_path = in_path.with_name(in_path.stem + "-filtered.csv")

    # Read with first column as sample IDs
    X = pd.read_csv(in_path, index_col=0)

    # Keep only *.sorted.raw
    X = X[X.index.astype(str).str.endswith(".sorted.raw")]

    # Strip suffix to get bare accession (e.g., ERR228024)
    X.index = X.index.str.replace(r"\.sorted\.raw$", "", regex=True)

    # If multiple rows collapse to the same accession, keep the max per feature
    if X.index.duplicated().any():
        X = X.groupby(level=0).max()

    X.to_csv(out_path)
    print(f"Wrote {out_path} with shape {X.shape}")

if __name__ == "__main__":
    main()
