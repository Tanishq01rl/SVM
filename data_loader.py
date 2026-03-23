import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def generate_dataset(
    n_samples: int = 12_000,
    n_features: int = 20,
    n_informative: int = 10,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=n_informative, n_redundant=4,
        n_clusters_per_class=2, flip_y=0.02, random_state=seed,
    )
    y = (2 * y - 1).astype(np.float32)                # {0,1} -> {-1,+1}
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    return (
        scaler.fit_transform(X_tr).astype(np.float32),
        scaler.transform(X_te).astype(np.float32),
        y_tr, y_te,
    )


class BatchLoader:

    def __init__(self, X, y, batch_size: int = 128, shuffle: bool = True, seed: int = 0):
        self.X, self.y = X, y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return (len(self.X) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.X)
        idx = self._rng.permutation(n) if self.shuffle else np.arange(n)
        for s in range(0, n, self.batch_size):
            sl = idx[s : s + self.batch_size]
            yield self.X[sl], self.y[sl]
