
import numpy as np


class SOM:
    """Self-Organizing Map for RGB color quantization."""

    def __init__(
        self,
        rows: int,
        cols: int,
        learning_rate: float = 0.5,
        sigma: float | None = None,
        seed: int = 42,
    ) -> None:
        if rows < 1 or cols < 1:
            raise ValueError("Grid dimensions must be positive.")
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")

        self.rows = rows
        self.cols = cols
        self.n_neurons = rows * cols
        self.learning_rate = learning_rate
        self.sigma = sigma if sigma is not None else max(rows, cols) / 2
        self.rng = np.random.default_rng(seed)
        self.weights: np.ndarray | None = None

        grid = np.array(
            [(r, c) for r in range(rows) for c in range(cols)],
            dtype=np.float64,
        )
        self.positions = grid

    def fit(self, data: np.ndarray, epochs: int = 20) -> "SOM":
        """Train the map using randomly sampled observations."""
        data = np.asarray(data, dtype=np.float64)

        if data.ndim != 2 or data.shape[0] == 0:
            raise ValueError("Data must be a non-empty 2D array.")
        if epochs < 1:
            raise ValueError("Epochs must be positive.")

        indices = self.rng.choice(
            len(data), size=self.n_neurons, replace=len(data) < self.n_neurons
        )
        self.weights = data[indices].copy()

        initial_sigma = max(self.sigma, 1e-12)

        for epoch in range(epochs):
            progress = epoch / max(epochs - 1, 1)
            learning_rate = self.learning_rate * (1 - progress)
            sigma = max(initial_sigma * (1 - progress), 1e-12)

            for index in self.rng.permutation(len(data)):
                sample = data[index]

                distances = np.linalg.norm(self.weights - sample, axis=1)
                bmu = int(np.argmin(distances))

                grid_distances = np.sum(
                    (self.positions - self.positions[bmu]) ** 2, axis=1
                )
                neighborhood = np.exp(-grid_distances / (2 * sigma**2))

                self.weights += (
                    learning_rate
                    * neighborhood[:, np.newaxis]
                    * (sample - self.weights)
                )

        return self

   
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Return the best matching unit index for each observation."""
        if self.weights is None:
            raise RuntimeError("Call fit before predict.")

        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != self.weights.shape[1]:
            raise ValueError("Data has an incompatible shape.")

        # Process observations in batches to limit memory consumption.
        assignments = np.empty(len(data), dtype=int)
        batch_size = 8192

        for start in range(0, len(data), batch_size):
            batch = data[start:start + batch_size]

            # Squared Euclidean distances: ||x-w||² = ||x||² + ||w||² - 2x·w
            distances = (
                np.sum(batch**2, axis=1, keepdims=True)
                + np.sum(self.weights**2, axis=1)[None, :]
                - 2 * batch @ self.weights.T
            )

            assignments[start:start + len(batch)] = np.argmin(
                distances, axis=1
            )

        return assignments
    

    def predict_two(self, data: np.ndarray) -> np.ndarray:
        """Return the indices of the two nearest neurons for each sample."""
        if self.weights is None:
            raise RuntimeError("Call fit before predict_two.")

        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != self.weights.shape[1]:
            raise ValueError("Data has an incompatible shape.")
        if len(self.weights) < 2:
            raise ValueError("At least two neurons are required.")

        assignments = np.empty((len(data), 2), dtype=int)
        batch_size = 8192
        weight_norms = np.sum(self.weights**2, axis=1)[None, :]

        for start in range(0, len(data), batch_size):
            batch = data[start:start + batch_size]
            distances = (
                np.sum(batch**2, axis=1, keepdims=True)
                + weight_norms
                - 2 * batch @ self.weights.T
            )
            assignments[start:start + len(batch)] = np.argsort(
                distances, axis=1
            )[:, :2]

        return assignments

    def topological_neighbors(self) -> list[set[int]]:
        """Return the four-connected neighborhood of the SOM grid."""
        neighbors = [set() for _ in range(self.n_neurons)]

        for i, (row, col) in enumerate(self.positions.astype(int)):
            if row > 0:
                neighbors[i].add(i - self.cols)
            if row < self.rows - 1:
                neighbors[i].add(i + self.cols)
            if col > 0:
                neighbors[i].add(i - 1)
            if col < self.cols - 1:
                neighbors[i].add(i + 1)

        return neighbors


    def transform(self, data: np.ndarray) -> np.ndarray:
        """Quantize observations to their best matching unit prototypes."""
        if self.weights is None:
            raise RuntimeError("Call fit before transform.")

        assignments = self.predict(data)
        return self.weights[assignments]
