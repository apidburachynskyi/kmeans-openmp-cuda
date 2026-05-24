import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import naive_kmeans
import parallel_kmeans


def make_data(n, d, true_k=10):
    X = np.empty((n, d), dtype=np.float32)

    # Same formula as cuda/make_data_kernel.
    # Use integer arithmetic before converting to float32.
    idx = np.arange(n, dtype=np.int64)
    cluster = idx % true_k

    for j in range(d):
        noise = ((idx * 17 + j * 31) % 100).astype(np.float32) / 100.0
        X[:, j] = (
            4.0 * (cluster % 16).astype(np.float32)
            + 0.02 * np.float32(j)
            + noise
        )

    return X


def run_custom(name, fn, X, k, max_iter, tol):
    t0 = time.perf_counter()
    labels, centers, inertia, n_iter, reason = fn(X, k, max_iter, tol)
    t1 = time.perf_counter()

    return {
        "implementation": name,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "clusters": k,
        "max_iter": max_iter,
        "iterations": n_iter,
        "time_seconds": t1 - t0,
        "time_per_iteration": (t1 - t0) / n_iter,
        "inertia": inertia,
        "stop_reason": reason,
    }


def run_sklearn(X, k, max_iter, tol):
    init = X[:k].copy()

    t0 = time.perf_counter()
    model = KMeans(
        n_clusters=k,
        init=init,
        n_init=1,
        max_iter=max_iter,
        tol=tol,
        algorithm="lloyd",
        random_state=0,
    )
    model.fit(X)
    t1 = time.perf_counter()

    return {
        "implementation": "sklearn",
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "clusters": k,
        "max_iter": max_iter,
        "iterations": int(model.n_iter_),
        "time_seconds": t1 - t0,
        "time_per_iteration": (t1 - t0) / int(model.n_iter_),
        "inertia": float(model.inertia_),
        "stop_reason": "sklearn_convergence",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=str, default="1000000,10000000")
    parser.add_argument("--clusters", type=str, default="2,4,8,10")
    parser.add_argument("--features", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=30, help="maximum iterations")
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--no-naive", action="store_true")
    parser.add_argument("--no-sklearn", action="store_true")
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    parallel_kmeans.set_threads(args.threads)

    sizes = [int(x) for x in args.sizes.split(",")]
    clusters = [int(x) for x in args.clusters.split(",")]

    rows = []

    for n in sizes:
        print(f"\nGenerating dataset n={n}, d={args.features}")
        X = make_data(n, args.features, true_k=10)

        for k in clusters:
            print(f"\n---- n={n}, k={k} ----")

            if not args.no_naive:
                row = run_custom("naive_numpy", naive_kmeans.fit_kmeans, X, k, args.iterations, args.tol)
                print(row)
                rows.append(row)

            row = run_custom("parallel_openmp", parallel_kmeans.fit_kmeans_parallel, X, k, args.iterations, args.tol)
            print(row)
            rows.append(row)

            if not args.no_sklearn:
                row = run_sklearn(X, k, args.iterations, args.tol)
                print(row)
                rows.append(row)

        del X

    df = pd.DataFrame(rows)

    naive_times = (
        df[df["implementation"] == "naive_numpy"]
        .set_index(["n_samples", "clusters"])["time_seconds"]
        .to_dict()
    )

    parallel_times = (
        df[df["implementation"] == "parallel_openmp"]
        .set_index(["n_samples", "clusters"])["time_seconds"]
        .to_dict()
    )

    df["speedup_vs_naive_total"] = [
        naive_times.get((r.n_samples, r.clusters), np.nan) / r.time_seconds
        for r in df.itertuples()
    ]

    df["speedup_vs_parallel_total"] = [
        parallel_times.get((r.n_samples, r.clusters), np.nan) / r.time_seconds
        for r in df.itertuples()
    ]

    naive_iter = (
        df[df["implementation"] == "naive_numpy"]
        .set_index(["n_samples", "clusters"])["time_per_iteration"]
        .to_dict()
    )

    parallel_iter = (
        df[df["implementation"] == "parallel_openmp"]
        .set_index(["n_samples", "clusters"])["time_per_iteration"]
        .to_dict()
    )

    df["speedup_vs_naive_per_iter"] = [
        naive_iter.get((r.n_samples, r.clusters), np.nan) / r.time_per_iteration
        for r in df.itertuples()
    ]

    df["speedup_vs_parallel_per_iter"] = [
        parallel_iter.get((r.n_samples, r.clusters), np.nan) / r.time_per_iteration
        for r in df.itertuples()
    ]

    os.makedirs("results", exist_ok=True)

    csv_path = "results/cpu_results.csv"
    xlsx_path = "results/cpu_results.xlsx"

    df.to_csv(csv_path, index=False)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="all_cpu_results")

        time_pivot = df.pivot_table(
            index=["n_samples", "clusters"],
            columns="implementation",
            values="time_seconds",
            aggfunc="first",
        ).reset_index()
        time_pivot.to_excel(writer, index=False, sheet_name="total_times")

        iter_pivot = df.pivot_table(
            index=["n_samples", "clusters"],
            columns="implementation",
            values="time_per_iteration",
            aggfunc="first",
        ).reset_index()
        iter_pivot.to_excel(writer, index=False, sheet_name="per_iteration")

        niter_pivot = df.pivot_table(
            index=["n_samples", "clusters"],
            columns="implementation",
            values="iterations",
            aggfunc="first",
        ).reset_index()
        niter_pivot.to_excel(writer, index=False, sheet_name="iterations")

    print(f"\nSaved {csv_path}")
    print(f"Saved {xlsx_path}")


if __name__ == "__main__":
    main()
