import numpy as np


class PegasosSVM:
    def __init__(self, n_features: int, C: float = 1.0):
        self.C = float(C)
        self.lam = 1.0 / self.C
        self.w = np.zeros(n_features, dtype=np.float32)
        self._t = 0

    def partial_fit(self, Z: np.ndarray, y: np.ndarray) -> float:
        Z, y = Z.astype(np.float32), y.astype(np.float32)
        self._t += 1
        eta = np.float32(1.0 / (self.lam * self._t))
        margins = y * (Z @ self.w)          # (batch,)
        active  = margins < 1.0

        self.w *= np.float32(1.0 - eta * self.lam)
        if active.any():
            self.w += (eta / Z.shape[0]) * (y[active] @ Z[active])

        # Project onto feasible ball (convergence guarantee)
        norm = float(np.linalg.norm(self.w))
        max_norm = 1.0 / np.sqrt(self.lam)
        if norm > max_norm:
            self.w *= np.float32(max_norm / norm)

        hinge = float(np.mean(np.maximum(np.float32(0.0), np.float32(1.0) - margins)))
        return 0.5 * self.lam * float(np.dot(self.w, self.w)) + hinge

    def decision_function(self, Z: np.ndarray) -> np.ndarray:
        return Z.astype(np.float32) @ self.w

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return np.sign(self.decision_function(Z)).astype(np.float32)

    def score(self, Z: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(Z) == y.astype(np.float32)))
