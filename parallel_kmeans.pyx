# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
from cython.parallel cimport prange
from openmp cimport omp_get_max_threads, omp_get_thread_num, omp_set_num_threads


def set_threads(int n):
    omp_set_num_threads(n)


def scaled_tol(X, double tol):
    return float(np.mean(np.var(X, axis=0)) * tol)


cdef inline void nearest_center(
    float[:, ::1] X,
    float[:, ::1] centers,
    Py_ssize_t i,
    Py_ssize_t d,
    Py_ssize_t k,
    int* label,
    float* distance
) noexcept nogil:
    cdef Py_ssize_t c, j
    cdef int best_c = 0
    cdef float best = 3.402823e38
    cdef float cur
    cdef float diff

    for c in range(k):
        cur = 0.0
        for j in range(d):
            diff = X[i, j] - centers[c, j]
            cur += diff * diff

        if cur < best:
            best = cur
            best_c = <int>c

    label[0] = best_c
    distance[0] = best


def assign_points_parallel(float[:, ::1] X, float[:, ::1] centers):
    cdef Py_ssize_t n = X.shape[0]
    cdef Py_ssize_t d = X.shape[1]
    cdef Py_ssize_t k = centers.shape[0]

    cdef cnp.ndarray[cnp.int32_t, ndim=1] labels = np.empty(n, dtype=np.int32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] distances = np.empty(n, dtype=np.float32)

    cdef int[::1] lab = labels
    cdef float[::1] dist = distances

    cdef Py_ssize_t i
    cdef double inertia = 0.0

    for i in prange(n, nogil=True, schedule="static"):
        nearest_center(X, centers, i, d, k, &lab[i], &dist[i])

    for i in range(n):
        inertia += dist[i]

    return labels, inertia


def update_centers_parallel(float[:, ::1] X, int[::1] labels, int k, float[:, ::1] old_centers):
    cdef Py_ssize_t n = X.shape[0]
    cdef Py_ssize_t d = X.shape[1]
    cdef int nt = omp_get_max_threads()

    cdef cnp.ndarray[cnp.float64_t, ndim=3] sums = np.zeros((nt, k, d), dtype=np.float64)
    cdef cnp.ndarray[cnp.int64_t, ndim=2] counts = np.zeros((nt, k), dtype=np.int64)
    cdef cnp.ndarray[cnp.float32_t, ndim=2] centers = np.zeros((k, d), dtype=np.float32)

    cdef double[:, :, ::1] local_sums = sums
    cdef long long[:, ::1] local_counts = counts
    cdef float[:, ::1] cen = centers

    cdef Py_ssize_t i, j
    cdef int t, c

    for i in prange(n, nogil=True, schedule="static"):
        t = omp_get_thread_num()
        c = labels[i]
        local_counts[t, c] += 1

        for j in range(d):
            local_sums[t, c, j] += X[i, j]

    for t in range(1, nt):
        for c in range(k):
            local_counts[0, c] += local_counts[t, c]
            for j in range(d):
                local_sums[0, c, j] += local_sums[t, c, j]

    for c in range(k):
        if local_counts[0, c] == 0:
            for j in range(d):
                cen[c, j] = old_centers[c, j]
        else:
            for j in range(d):
                cen[c, j] = <float>(local_sums[0, c, j] / local_counts[0, c])

    return centers


def fit_kmeans_parallel(X, int k, int max_iter=30, double tol=1e-4):
    X = np.ascontiguousarray(X, dtype=np.float32)
    centers = np.ascontiguousarray(X[:k].copy(), dtype=np.float32)

    cdef double tol_value = scaled_tol(X, tol)
    cdef double shift
    cdef int it
    old_labels = None
    reason = "max_iter"

    for it in range(max_iter):
        labels, inertia = assign_points_parallel(X, centers)
        new_centers = update_centers_parallel(X, labels, k, centers)
        shift = float(np.sum((new_centers - centers) ** 2))
        centers = np.ascontiguousarray(new_centers, dtype=np.float32)

        if old_labels is not None and np.array_equal(labels, old_labels):
            reason = "labels_unchanged"
            break

        if shift <= tol_value:
            reason = "center_shift"
            break

        old_labels = labels.copy()

    labels, inertia = assign_points_parallel(X, centers)
    return labels, centers, float(inertia), it + 1, reason
