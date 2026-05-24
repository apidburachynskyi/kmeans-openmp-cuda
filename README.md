# K-means large benchmark

Files:

- `naive_kmeans.py`: sequential NumPy baseline
- `parallel_kmeans.pyx`: Cython/OpenMP version
- `benchmark.py`: CPU benchmark with sklearn comparison
- `cuda/kmeans_cuda.cu`: CUDA benchmark
- `combine_results.py`: combines CPU and CUDA outputs into one Excel file

## CPU

```powershell
pip install -r requirements.txt
python setup.py build_ext --inplace
python benchmark.py --sizes 1000000,10000000 --clusters 2,4,8,10 --features 16 --iterations 30 --threads 8
```

## CUDA

RTX 5070 Ti:

```powershell
cd cuda
nvcc -O3 -arch=sm_120 kmeans_cuda.cu -o kmeans_cuda.exe
mkdir ..\results
.\kmeans_cuda.exe 16 30 > ..\results\cuda_results.csv
cd ..
```