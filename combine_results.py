import argparse
import os
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=str, default="results/cpu_results.csv")
    parser.add_argument("--cuda", type=str, default="results/cuda_results.csv")
    parser.add_argument("--out", type=str, default="results/all_results.xlsx")
    args = parser.parse_args()

    cpu = pd.read_csv(args.cpu)
    frames = [cpu]

    if os.path.exists(args.cuda):
        cuda = pd.read_csv(args.cuda)
        frames.append(cuda)
    else:
        print(f"CUDA file not found: {args.cuda}")
        print("Only CPU results will be written.")

    df = pd.concat(frames, ignore_index=True)

    parallel_times = (
        df[df["implementation"] == "parallel_openmp"]
        .set_index(["n_samples", "clusters"])["time_seconds"]
        .to_dict()
    )

    naive_times = (
        df[df["implementation"] == "naive_numpy"]
        .set_index(["n_samples", "clusters"])["time_seconds"]
        .to_dict()
    )

    parallel_iter = (
        df[df["implementation"] == "parallel_openmp"]
        .set_index(["n_samples", "clusters"])["time_per_iteration"]
        .to_dict()
    )

    naive_iter = (
        df[df["implementation"] == "naive_numpy"]
        .set_index(["n_samples", "clusters"])["time_per_iteration"]
        .to_dict()
    )

    df["speedup_vs_parallel_total"] = [
        parallel_times.get((r.n_samples, r.clusters), np.nan) / r.time_seconds
        for r in df.itertuples()
    ]

    df["speedup_vs_naive_total"] = [
        naive_times.get((r.n_samples, r.clusters), np.nan) / r.time_seconds
        for r in df.itertuples()
    ]

    df["speedup_vs_parallel_per_iter"] = [
        parallel_iter.get((r.n_samples, r.clusters), np.nan) / r.time_per_iteration
        for r in df.itertuples()
    ]

    df["speedup_vs_naive_per_iter"] = [
        naive_iter.get((r.n_samples, r.clusters), np.nan) / r.time_per_iteration
        for r in df.itertuples()
    ]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="all_results")

        df.pivot_table(
            index=["n_samples", "clusters"],
            columns="implementation",
            values="time_seconds",
            aggfunc="first",
        ).reset_index().to_excel(writer, index=False, sheet_name="total_times")

        df.pivot_table(
            index=["n_samples", "clusters"],
            columns="implementation",
            values="time_per_iteration",
            aggfunc="first",
        ).reset_index().to_excel(writer, index=False, sheet_name="per_iteration")

        df.pivot_table(
            index=["n_samples", "clusters"],
            columns="implementation",
            values="iterations",
            aggfunc="first",
        ).reset_index().to_excel(writer, index=False, sheet_name="iterations")

    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
