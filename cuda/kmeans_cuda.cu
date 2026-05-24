/*
 * K-means CUDA benchmark
 * Authors : Arsen Pidburachynskyi
 * Initial code : Lokman A. Abbas-Turki

 * The program prints CSV results.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

void testCUDA(cudaError_t error, const char *file, int line)
{
    if (error != cudaSuccess)
    {
        printf("There is an error in file %s at line %d: %s\n",
               file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
#define testCUDA(error) (testCUDA(error, __FILE__, __LINE__))

__global__ void make_data_k(float *X, int n, int d, int true_k)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += step)
    {
        int c = i % true_k;

        for (int j = 0; j < d; j++)
        {
            float noise = ((i * 17 + j * 31) % 100) / 100.0f;
            X[i * d + j] = 4.0f * (c % 16) + 0.02f * j + noise;
        }
    }
}

__global__ void init_centers_k(float *centers, const float *X, int k, int d)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = k * d;

    if (idx < total)
        centers[idx] = X[idx];
}

__global__ void assign_k(const float *X, const float *centers,
                         int *labels, int *old_labels, int *changed,
                         float *distances, int n, int d, int k)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += step)
    {
        int best_cluster = 0;
        float best_dist = 3.402823e38f;

        for (int c = 0; c < k; c++)
        {
            float dist = 0.0f;

            for (int j = 0; j < d; j++)
            {
                float diff = X[i * d + j] - centers[c * d + j];
                dist += diff * diff;
            }

            if (dist < best_dist)
            {
                best_dist = dist;
                best_cluster = c;
            }
        }

        labels[i] = best_cluster;
        distances[i] = best_dist;

        if (old_labels[i] != best_cluster)
            atomicExch(changed, 1);
    }
}

__global__ void update_sums_k(const float *X, const int *labels,
                              float *sums, int *counts,
                              int n, int d, int k)
{
    extern __shared__ unsigned char A[];

    float *local_sums = (float *)A;
    int *local_counts = (int *)&local_sums[k * d];

    for (int i = threadIdx.x; i < k * d; i += blockDim.x)
        local_sums[i] = 0.0f;

    for (int i = threadIdx.x; i < k; i += blockDim.x)
        local_counts[i] = 0;

    __syncthreads();

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += step)
    {
        int c = labels[i];

        atomicAdd(&local_counts[c], 1);

        for (int j = 0; j < d; j++)
            atomicAdd(&local_sums[c * d + j], X[i * d + j]);
    }

    __syncthreads();

    for (int i = threadIdx.x; i < k * d; i += blockDim.x)
        atomicAdd(&sums[i], local_sums[i]);

    for (int i = threadIdx.x; i < k; i += blockDim.x)
        atomicAdd(&counts[i], local_counts[i]);
}

__global__ void normalize_centers_k(float *centers,
                                    const float *sums, const int *counts,
                                    int d, int k)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = k * d;

    if (idx < total)
    {
        int c = idx / d;

        if (counts[c] > 0)
            centers[idx] = sums[idx] / counts[c];
    }
}


double sum_inertia(const float *distances, int n)
{
    double s = 0.0;

    for (int i = 0; i < n; i++)
        s += distances[i];

    return s;
}

static void run_and_time(int n, int d, int k, int max_iter,
                         int NB, int NTPB,
                         int *out_iter, float *out_seconds,
                         float *out_seconds_iter, double *out_inertia,
                         const char **out_reason)
{
    size_t X_size = (size_t)n * d * sizeof(float);
    size_t centers_size = (size_t)k * d * sizeof(float);
    size_t distances_size = (size_t)n * sizeof(float);
    size_t labels_size = (size_t)n * sizeof(int);
    size_t shared_size = (size_t)k * d * sizeof(float) + (size_t)k * sizeof(int);

    float *XGPU, *CentersGPU, *DistancesGPU, *SumsGPU;
    int *LabelsGPU, *OldLabelsGPU, *CountsGPU, *ChangedGPU;

    testCUDA(cudaMalloc(&XGPU, X_size));
    testCUDA(cudaMalloc(&CentersGPU, centers_size));
    testCUDA(cudaMalloc(&DistancesGPU, distances_size));
    testCUDA(cudaMalloc(&SumsGPU, centers_size));
    testCUDA(cudaMalloc(&LabelsGPU, labels_size));
    testCUDA(cudaMalloc(&OldLabelsGPU, labels_size));
    testCUDA(cudaMalloc(&CountsGPU, (size_t)k * sizeof(int)));
    testCUDA(cudaMalloc(&ChangedGPU, sizeof(int)));

    make_data_k<<<NB, NTPB>>>(XGPU, n, d, 10);
    testCUDA(cudaGetLastError());

    init_centers_k<<<(k * d + NTPB - 1) / NTPB, NTPB>>>(CentersGPU, XGPU, k, d);
    testCUDA(cudaGetLastError());

    testCUDA(cudaMemset(OldLabelsGPU, 0xFF, labels_size));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    int actual_iter = 0;
    int changed = 1;
    const char *reason = "max_iter";

    for (int it = 0; it < max_iter; it++)
    {
        changed = 0;
        testCUDA(cudaMemcpy(ChangedGPU, &changed, sizeof(int), cudaMemcpyHostToDevice));

        assign_k<<<NB, NTPB>>>(XGPU, CentersGPU, LabelsGPU, OldLabelsGPU,
                               ChangedGPU, DistancesGPU, n, d, k);
        testCUDA(cudaGetLastError());

        testCUDA(cudaMemcpy(&changed, ChangedGPU, sizeof(int), cudaMemcpyDeviceToHost));

        actual_iter = it + 1;

        if (changed == 0)
        {
            reason = "labels_unchanged";
            break;
        }

        testCUDA(cudaMemset(SumsGPU, 0, centers_size));
        testCUDA(cudaMemset(CountsGPU, 0, (size_t)k * sizeof(int)));

        update_sums_k<<<NB, NTPB, shared_size>>>(XGPU, LabelsGPU, SumsGPU,
                                                 CountsGPU, n, d, k);
        testCUDA(cudaGetLastError());

        normalize_centers_k<<<(k * d + NTPB - 1) / NTPB, NTPB>>>(CentersGPU,
                                                                 SumsGPU,
                                                                 CountsGPU,
                                                                 d, k);
        testCUDA(cudaGetLastError());

        int *tmp = OldLabelsGPU;
        OldLabelsGPU = LabelsGPU;
        LabelsGPU = tmp;
    }

    assign_k<<<NB, NTPB>>>(XGPU, CentersGPU, LabelsGPU, OldLabelsGPU,
                           ChangedGPU, DistancesGPU, n, d, k);
    testCUDA(cudaGetLastError());

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    float *DistancesCPU = (float *)malloc(distances_size);
    testCUDA(cudaMemcpy(DistancesCPU, DistancesGPU, distances_size, cudaMemcpyDeviceToHost));

    *out_iter = actual_iter;
    *out_seconds = ms / 1000.0f;
    *out_seconds_iter = *out_seconds / actual_iter;
    *out_inertia = sum_inertia(DistancesCPU, n);
    *out_reason = reason;

    free(DistancesCPU);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(XGPU);
    cudaFree(CentersGPU);
    cudaFree(DistancesGPU);
    cudaFree(SumsGPU);
    cudaFree(LabelsGPU);
    cudaFree(OldLabelsGPU);
    cudaFree(CountsGPU);
    cudaFree(ChangedGPU);
}

int main(int argc, char **argv)
{
    int d = 16;
    int max_iter = 30;

    if (argc > 1)
        d = atoi(argv[1]);

    if (argc > 2)
        max_iter = atoi(argv[2]);

    int NTPB = 256;
    int sizes[] = {1000000, 10000000};
    int clusters[] = {2, 4, 8, 10};

    int nb_sizes = 2;
    int nb_clusters = 4;

    printf("implementation,n_samples,n_features,clusters,max_iter,iterations,time_seconds,time_per_iteration,inertia,stop_reason\n");

    for (int i = 0; i < nb_sizes; i++)
    {
        for (int j = 0; j < nb_clusters; j++)
        {
            int n = sizes[i];
            int k = clusters[j];

            int NB = (n + NTPB - 1) / NTPB;
            if (NB > 4096)
                NB = 4096;

            int actual_iter;
            float seconds, seconds_iter;
            double inertia;
            const char *reason;

            run_and_time(n, d, k, max_iter, NB, NTPB,
                         &actual_iter, &seconds, &seconds_iter,
                         &inertia, &reason);

            printf("cuda,%d,%d,%d,%d,%d,%.6f,%.6f,%.3f,%s\n",
                   n, d, k, max_iter, actual_iter,
                   seconds, seconds_iter, inertia, reason);
        }
    }

    return 0;
}
