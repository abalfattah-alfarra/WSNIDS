# 01_merge_csv.py
import argparse, pandas as pd, pathlib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsnds", required=True, help="Path to wsn-ds CSV")
    ap.add_argument("--out",    required=True, help="Output Parquet file")
    args = ap.parse_args()

    csv_path  = pathlib.Path(args.wsnds)
    out_path  = pathlib.Path(args.out)

    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print("Rows:", len(df), "Cols:", len(df.columns))

    print(f"Writing Parquet to {out_path} ...")
    df.to_parquet(out_path, index=False)
    print("Done ✓")

if __name__ == "__main__":
    main()
