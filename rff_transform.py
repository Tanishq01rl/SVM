import numpy as np


class RFFTransformer:
    def __init__(self, D: int, sigma: float = 1.0, seed: int = 42):
        self.D, self.sigma, self.seed = D, sigma, seed
        self.W: np.ndarray | None = None   # (D, d)
        self.b: np.ndarray | None = None   # (D,)

    def fit(self, n_input_features: int) -> "RFFTransformer":
        """Sample frequency matrix W and bias b."""
        rng = np.random.default_rng(self.seed)
        self.W = rng.normal(0.0, 1.0 / self.sigma,
                            size=(self.D, n_input_features)).astype(np.float32)
        self.b = rng.uniform(0.0, 2.0 * np.pi, size=(self.D,)).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """X: (n, d) -> Z: (n, D).  No kernel matrix is ever stored."""
        if self.W is None:
            raise RuntimeError("Call fit() before transform().")
        projection = X.astype(np.float32) @ self.W.T + self.b  # (n, D)
        return np.float32(np.sqrt(2.0 / self.D)) * np.cos(projection)
