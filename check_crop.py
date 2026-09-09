"""
Test crop bounds on a single exported Fluent CSV.

Reports the domain extents, applies a candidate crop box, and reports how many
nodes survive and whether the field ranges are retained. Run this on a couple of
design points before committing to bounds for the whole dataset.

Usage:
    python check_crop.py path/to/dp0001.csv
    python check_crop.py path/to/dp0001.csv --xmin -1 --xmax 3 --ymin -0.5 --ymax 2.5
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--xmin", type=float, default=-1.0)
    ap.add_argument("--xmax", type=float, default=3.0)
    ap.add_argument("--ymin", type=float, default=-0.5)
    ap.add_argument("--ymax", type=float, default=2.5)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = [c.strip() for c in df.columns]

    print("columns:", list(df.columns))
    print(f"rows: {len(df):,}\n")

    # The export produced duplicate coordinate columns; pandas suffixes repeats
    # as name.1, name.2 and so on, so the plain name is the first occurrence.
    def col(*candidates: str) -> str:
        for c in candidates:
            if c in df.columns:
                return c
        raise KeyError(f"none of {candidates} found in {list(df.columns)}")

    xc = col("x-coordinate", "x-coord", "x")
    yc = col("y-coordinate", "y-coord", "y")
    pc = col("pressure", "static-pressure", "p")
    vc = col("velocity-magnitude", "velocity magnitude", "v")
    volc = col("cell-volume", "cell volume", "volume")

    print("full domain")
    print(f"  x        {df[xc].min():>12.4g}  to {df[xc].max():>12.4g}")
    print(f"  y        {df[yc].min():>12.4g}  to {df[yc].max():>12.4g}")
    print(f"  pressure {df[pc].min():>12.4g}  to {df[pc].max():>12.4g}")
    print(f"  velocity {df[vc].min():>12.4g}  to {df[vc].max():>12.4g}")
    print(f"  cell vol {df[volc].min():>12.4g}  to {df[volc].max():>12.4g}\n")

    mask = (
        df[xc].between(args.xmin, args.xmax)
        & df[yc].between(args.ymin, args.ymax)
    )
    cropped = df[mask]

    if len(cropped) == 0:
        print("CROP IS EMPTY -- the bounds do not intersect the domain.")
        return

    print(f"crop  x in [{args.xmin}, {args.xmax}]  y in [{args.ymin}, {args.ymax}]")
    print(f"  nodes    {len(cropped):>12,}  ({100 * len(cropped) / len(df):.1f}% of {len(df):,})")
    print(f"  pressure {cropped[pc].min():>12.4g}  to {cropped[pc].max():>12.4g}")
    print(f"  velocity {cropped[vc].min():>12.4g}  to {cropped[vc].max():>12.4g}")
    print(f"  cell vol {cropped[volc].min():>12.4g}  to {cropped[volc].max():>12.4g}\n")

    # Retaining the field extremes matters: losing them means the crop has cut
    # into the suction peak or the stagnation region.
    for label, c in (("pressure", pc), ("velocity", vc)):
        full = df[c].max() - df[c].min()
        kept = cropped[c].max() - cropped[c].min()
        pct = 100 * kept / full if full else 100.0
        flag = "" if pct > 98 else "   <-- CHECK: range lost"
        print(f"  {label} range retained: {pct:5.1f}%{flag}")

    if not args.no_plot:
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ax[0].scatter(df[xc], df[yc], s=0.2, c=df[vc], cmap="viridis")
        ax[0].add_patch(
            plt.Rectangle(
                (args.xmin, args.ymin),
                args.xmax - args.xmin,
                args.ymax - args.ymin,
                fill=False, edgecolor="red", linewidth=1.5,
            )
        )
        ax[0].set_title(f"full domain ({len(df):,} nodes)")
        ax[1].scatter(cropped[xc], cropped[yc], s=0.5, c=cropped[vc], cmap="viridis")
        ax[1].set_title(f"cropped ({len(cropped):,} nodes)")
        for a in ax:
            a.set_xlabel("x")
            a.set_ylabel("y")
            a.set_aspect("equal")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()