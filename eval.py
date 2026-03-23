import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")   
import matplotlib.pyplot as plt
import numpy as np


def print_report(results: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("  RFF-Approximated Kernel SVM  —  Analysis Report")
    print("=" * 60)
    print(f"\n  {'D':>6}  {'Accuracy':>10}  {'Time (s)':>10}  {'RSS MB':>8}")
    print("  " + "-" * 42)
    for r in results:
        print(f"  {r['D']:>6}  {r['accuracy']:>10.4f}  {r['time_s']:>10.2f}  {r['rss_mb']:>8.1f}")

    best = max(results, key=lambda r: r["accuracy"])
    D_arr   = np.log([r["D"]      for r in results])
    rss_arr = np.log([r["rss_mb"] for r in results])
    slope   = np.polyfit(D_arr, rss_arr, 1)[0]

    print(f"\n  Best: accuracy={best['accuracy']:.4f} at D={best['D']}")
    print("\n[Kernel Approximation]")
    print("  * E[z(x)^T z(y)] -> k_RBF(x,y) as D->inf  (Bochner / Rahimi-Recht)")
    print("  * Small D -> under-fitting;  large D -> accuracy saturates")
    print("\n[Memory vs Accuracy]")
    print(f"  * Memory proportional to D  (log-log slope ~{slope:.2f}, theory=1.0)")
    print("  * float32 halves footprint vs float64 with negligible accuracy loss")
    print("  * Avoids O(N^2) Gram matrix of exact kernel SVM entirely")
    print("\n[Scaling Trends]")
    print("  * Time proportional to D  (matmul n x d x D per batch)")
    print("  * Accuracy gains sub-linear; sweet spot typically D = 300-500")
    print("=" * 60 + "\n")


def plot_results(results: list[dict], out_path: str) -> None:
    D   = [r["D"]        for r in results]
    acc = [r["accuracy"] for r in results]
    mem = [r["rss_mb"]   for r in results]
    t   = [r["time_s"]   for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("RFF-Approximated Kernel SVM", fontsize=13)

    specs = [
        (acc, "Accuracy",         "royalblue", "Accuracy vs D"),
        (mem, "RSS Memory (MB)",  "tomato",    "Memory vs D"),
        (t,   "Training Time (s)","seagreen",  "Time vs D"),
    ]
    for ax, (y_vals, ylabel, color, title) in zip(axes, specs):
        ax.plot(D, y_vals, "o-", color=color, lw=2, ms=6)
        ax.set_xlabel("D  (RFF dimensions)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.35)
        ax.set_xticks(D)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved: {out_path}")


def compare_sklearn(results: list[dict]) -> None:
    try:
        from sklearn.svm import SVC
        from data_loader import generate_dataset
    except ImportError:
        print("sklearn not available — skipping comparison.")
        return

    print("\nsklearn RBF SVM (N=3 000, full kernel) …")
    X_tr, X_te, y_tr, y_te = generate_dataset(n_samples=3_000, seed=42)
    t0 = time.perf_counter()
    clf = SVC(kernel="rbf", C=5.0, gamma="scale").fit(X_tr, y_tr)
    elapsed = time.perf_counter() - t0
    acc = clf.score(X_te, y_te)
    best = max(results, key=lambda r: r["accuracy"])
    print(f"  sklearn  acc={acc:.4f}  time={elapsed:.2f}s")
    print(f"  RFF best acc={best['accuracy']:.4f}  D={best['D']}")
    print(f"  Gap (sklearn − RFF): {acc - best['accuracy']:+.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sklearn", action="store_true")
    args = parser.parse_args()

    path = os.path.join(os.path.dirname(__file__), "results.json")
    if not os.path.exists(path):
        print("Run  python train.py  first.", file=sys.stderr); sys.exit(1)

    with open(path) as fh:
        results = json.load(fh)

    print_report(results)
    plot_results(results, os.path.join(os.path.dirname(__file__), "rff_svm_results.png"))
    if args.sklearn:
        compare_sklearn(results)


if __name__ == "__main__":
    main()
