
import numpy as np


class GNG:
    """Growing Neural Gas for RGB color quantization."""

    def __init__(
        self,
        max_neurons: int = 64,
        learning_rate_bmu: float = 0.05,
        learning_rate_neighbors: float = 0.0006,
        max_age: int = 50,
        insertion_interval: int = 100,
        alpha: float = 0.5,
        beta: float = 0.0005,
        seed: int = 42,
    ) -> None:
        if max_neurons < 2:
            raise ValueError("GNG requires at least two neurons.")
        if learning_rate_bmu <= 0 or learning_rate_neighbors < 0:
            raise ValueError("Learning rates must be positive (neighbor rate may be zero).")
        if max_age < 0 or insertion_interval < 1:
            raise ValueError("Invalid age or insertion interval.")
        if not 0 < alpha <= 1 or not 0 < beta < 1:
            raise ValueError("alpha must be in (0, 1] and beta in (0, 1).")

        self.max_neurons = max_neurons
        self.learning_rate_bmu = learning_rate_bmu
        self.learning_rate_neighbors = learning_rate_neighbors
        self.max_age = max_age
        self.insertion_interval = insertion_interval
        self.alpha = alpha
        self.beta = beta
        self.rng = np.random.default_rng(seed)

        self.weights: np.ndarray | None = None
        self.edges: dict[tuple[int, int], int] = {}
        self.errors: np.ndarray | None = None
        self.step = 0

    @staticmethod
    def _edge(a: int, b: int) -> tuple[int, int]:
        return (min(a, b), max(a, b))

    def _connect(self, a: int, b: int) -> None:
        self.edges[self._edge(a, b)] = 0

    def _remove_node(self, node: int) -> None:
        self.edges = {
            edge: age for edge, age in self.edges.items()
            if node not in edge
        }
        self.weights = np.delete(self.weights, node, axis=0)
        self.errors = np.delete(self.errors, node)

        remapped = {}
        for (a, b), age in self.edges.items():
            new_a = a - (a > node)
            new_b = b - (b > node)
            remapped[self._edge(new_a, new_b)] = age
        self.edges = remapped

    def _insert_node(self) -> None:
        if len(self.weights) >= self.max_neurons:
            return

        incident = {}
        for (a, b), age in self.edges.items():
            incident.setdefault(a, set()).add(b)
            incident.setdefault(b, set()).add(a)

        candidates = [
            i for i in range(len(self.weights)) if incident.get(i)
        ]
        if not candidates:
            return

        q = max(candidates, key=lambda i: self.errors[i])
        f = max(incident[q], key=lambda i: self.errors[i])

        new_weight = (self.weights[q] + self.weights[f]) / 2
        new_index = len(self.weights)
        self.weights = np.vstack([self.weights, new_weight])
        self.errors = np.append(self.errors, self.errors[q] * 0.5)

        self.errors[q] *= 0.5
        self.errors[f] *= 0.5

        self.edges.pop(self._edge(q, f), None)
        self._connect(q, new_index)
        self._connect(f, new_index)

    def fit(self, data: np.ndarray, n_steps: int = 10000) -> "GNG":
        """Train GNG by repeatedly sampling observations."""
        data = np.asarray(data, dtype=np.float64)

        if data.ndim != 2 or len(data) < 2:
            raise ValueError("Data must contain at least two observations.")
        if n_steps < 1:
            raise ValueError("n_steps must be positive.")

        initial = self.rng.choice(len(data), size=2, replace=False)
        self.weights = data[initial].copy()
        self.errors = np.zeros(2, dtype=np.float64)
        self.edges = {}
        self._connect(0, 1)
        self.step = 0

        for step in range(1, n_steps + 1):
            self.step = step
            sample = data[self.rng.integers(len(data))]

            distances = np.linalg.norm(self.weights - sample, axis=1)
            order = np.argsort(distances)
            s1, s2 = int(order[0]), int(order[1])

            self.errors[s1] += distances[s1] ** 2

            # Age all edges connected to the best matching unit.
            for edge in list(self.edges):
                if s1 in edge:
                    self.edges[edge] += 1

            self.weights[s1] += self.learning_rate_bmu * (
                sample - self.weights[s1]
            )

            neighbors = [
                b if a == s1 else a
                for (a, b) in self.edges if s1 in (a, b)
            ]
            for neighbor in neighbors:
                self.weights[neighbor] += self.learning_rate_neighbors * (
                    sample - self.weights[neighbor]
                )

            self._connect(s1, s2)

            # Remove edges that have become too old.
            self.edges = {
                edge: age for edge, age in self.edges.items()
                if age <= self.max_age
            }

            # Remove isolated neurons.
            connected = {node for edge in self.edges for node in edge}
            for node in reversed(range(len(self.weights))):
                if node not in connected and len(self.weights) > 2:
                    self._remove_node(node)
                    connected = {n for edge in self.edges for n in edge}

            if step % self.insertion_interval == 0:
                self._insert_node()

            self.errors *= (1.0 - self.beta)

        return self

    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Return the nearest GNG prototype index for each observation."""
        if self.weights is None:
            raise RuntimeError("Call fit before predict.")

        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != self.weights.shape[1]:
            raise ValueError("Data has an incompatible shape.")

        assignments = np.empty(len(data), dtype=int)
        batch_size = 8192
        weight_norms = np.sum(self.weights**2, axis=1)[None, :]

        for start in range(0, len(data), batch_size):
            batch = data[start:start + batch_size]
            distances = (
                np.sum(batch**2, axis=1, keepdims=True)
                + weight_norms
                - 2 * batch @ self.weights.T
            )
            assignments[start:start + len(batch)] = np.argmin(
                distances, axis=1
            )

        return assignments

    
    def predict_two(self, data: np.ndarray) -> np.ndarray:
        """Return the indices of the two nearest neurons per sample."""
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
        """Return the current GNG adjacency list."""
        if self.weights is None:
            raise RuntimeError("Call fit before topological_neighbors.")

        neighbors = [set() for _ in range(len(self.weights))]

        for a, b in self.edges:
            neighbors[a].add(b)
            neighbors[b].add(a)

        return neighbors

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Quantize observations to their nearest GNG prototypes."""
        if self.weights is None:
            raise RuntimeError("Call fit before transform.")

        return self.weights[self.predict(data)]
