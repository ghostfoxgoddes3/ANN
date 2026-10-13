
import numpy as np


def quantization_error(
    data: np.ndarray,
    prototypes: np.ndarray,
    assignments: np.ndarray,
) -> float:
    """Return the mean Euclidean distance to assigned prototypes."""
    data = np.asarray(data, dtype=np.float64)
    prototypes = np.asarray(prototypes, dtype=np.float64)
    assignments = np.asarray(assignments, dtype=int)

    if data.ndim != 2 or prototypes.ndim != 2:
        raise ValueError("Data and prototypes must be 2D arrays.")
    if data.shape[1] != prototypes.shape[1]:
        raise ValueError("Data and prototypes must have the same feature count.")
    if assignments.shape != (data.shape[0],):
        raise ValueError("There must be one assignment per sample.")
    if len(prototypes) == 0:
        raise ValueError("At least one prototype is required.")
    if np.any(assignments < 0) or np.any(assignments >= len(prototypes)):
        raise ValueError("Assignments contain invalid prototype indices.")

    distances = np.linalg.norm(data - prototypes[assignments], axis=1)
    return float(np.mean(distances))


def topological_error(
    assignments: np.ndarray,
    neighbors: list[set[int]],
) -> float:
    """Return the fraction of samples whose two BMUs are not connected.

    `neighbors[i]` contains the indices of prototypes directly connected
    to prototype i. For a SOM, provide grid-adjacent prototypes.
    For a GNG, provide the current graph adjacency.
    """
    assignments = np.asarray(assignments, dtype=int)

    if assignments.ndim != 2 or assignments.shape[1] != 2:
        raise ValueError("Assignments must have shape (n_samples, 2).")
    if len(neighbors) == 0:
        raise ValueError("At least one prototype is required.")
    if np.any(assignments < 0) or np.any(assignments >= len(neighbors)):
        raise ValueError("Assignments contain invalid prototype indices.")

    if len(assignments) == 0:
        return 0.0

    errors = [
        second not in neighbors[first]
        for first, second in assignments
    ]
    return float(np.mean(errors))
