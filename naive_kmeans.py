import numpy as np


def initial_centers(X, k):
    return X[:k].copy()


def scaled_tol(X, tol):
    return float(np.mean(np.var(X, axis=0)) * tol)


def assign_points(X, centers, chunk_size=200_000):
    n = X.shape[0]
    k = centers.shape[0]

    labels = np.empty(n, dtype=np.int32)
    distances = np.empty(n, dtype=np.float32)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        Xc = X[start:end]

        best = np.full(end - start, np.inf, dtype=np.float32)
        lab = np.zeros(end - start, dtype=np.int32)

        for c in range(k):
            diff = Xc - centers[c]
            dist = np.sum(diff * diff, axis=1)

            mask = dist < best
            best[mask] = dist[mask]
            lab[mask] = c

        labels[start:end] = lab
        distances[start:end] = best

    return labels, float(distances.sum())


def update_centers(X, labels, k, old_centers):
    centers = old_centers.copy()

    for c in range(k):
        pts = X[labels == c]
        if len(pts) > 0:
            centers[c] = pts.mean(axis=0)

    return centers


def fit_kmeans(X, k, max_iter=30, tol=1e-4):
    centers = initial_centers(X, k)
    tol_value = scaled_tol(X, tol)
    old_labels = None
    reason = "max_iter"

    for it in range(max_iter):
        labels, inertia = assign_points(X, centers)
        new_centers = update_centers(X, labels, k, centers)
        shift = float(np.sum((new_centers - centers) ** 2))
        centers = new_centers

        if old_labels is not None and np.array_equal(labels, old_labels):
            reason = "labels_unchanged"
            break

        if shift <= tol_value:
            reason = "center_shift"
            break

        old_labels = labels.copy()

    labels, inertia = assign_points(X, centers)
    return labels, centers, inertia, it + 1, reason
