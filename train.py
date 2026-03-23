import json
import os
import time

import numpy as np
import psutil

from data_loader import BatchLoader, generate_dataset
from pegasos_svm import PegasosSVM
from rff_transform import RFFTransformer


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1_048_576


def train_one(
    D: int,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    *,
    n_epochs: int = 8, batch_size: int = 128,
    C: float = 5.0, sigma: float = 1.0, seed: int = 42,
) -> dict:
    rff = RFFTransformer(D=D, sigma=sigma, seed=seed).fit(X_train.shape[1])
    svm = PegasosSVM(n_features=D, C=C)
    loader = BatchLoader(X_train, y_train, batch_size=batch_size, seed=seed)

    rss_before = _rss_mb()
    t0 = time.perf_counter()

    for _ in range(n_epochs):
        for X_b, y_b in loader:
            Z_b = rff.transform(X_b)   # (batch, D) — never stored globally
            svm.partial_fit(Z_b, y_b)

    elapsed = time.perf_counter() - t0
    rss_peak = _rss_mb()

    # Chunked evaluation — keeps memory O(chunk × D)
    correct, n_te = 0, len(X_test)
    for s in range(0, n_te, 512):
        Zc = rff.transform(X_test[s : s + 512])
        correct += int(np.sum(svm.predict(Zc) == y_test[s : s + 512]))

    return {
        "D": D,
        "accuracy":      round(correct / n_te, 6),
        "time_s":        round(elapsed, 3),
        "rss_mb":        round(rss_peak, 2),
        "rss_delta_mb":  round(rss_peak - rss_before, 2),
    }


def main():
    print("Generating dataset …")
    X_tr, X_te, y_tr, y_te = generate_dataset(n_samples=12_000, seed=42)
    print(f"  train {X_tr.shape}  test {X_te.shape}  dtype={X_tr.dtype}")

    # sigma via sklearn "scale" heuristic: gamma=1/(d*var) -> sigma=sqrt(d/2)
    sigma = float(np.sqrt(X_tr.shape[1] / 2.0))
    print(f"  sigma (bandwidth) = {sigma:.3f}")

    results = []
    for D in [50, 100, 300, 500, 1000]:
        print(f"\n-- D={D} ----------------------------------")
        r = train_one(D, X_tr, y_tr, X_te, y_te, sigma=sigma)
        results.append(r)
        print(f"  acc={r['accuracy']:.4f}  time={r['time_s']:.2f}s  "
              f"RSS={r['rss_mb']:.1f} MB  (delta={r['rss_delta_mb']:+.1f})")

    out = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved: {out}\nRun:  python eval.py  (add --sklearn for comparison)")


if __name__ == "__main__":
    main()
