# 🧮 FFT Backend Benchmark (SciPy / FFTW / CuPy)

This repository provides a unified benchmark for comparing the performance of different FFT backends — **SciPy (PocketFFT)**, **pyFFTW**, and **CuPy (cuFFT)** — under consistent array and workload conditions.
The goal is to measure not only FFT performance, but also the impact of **element-wise arithmetic** and **reduction (sum)** operations that typically follow FFTs in scientific workloads.  
Through this, we can analyze how performance bottlenecks shift between CPU single-thread, CPU multi-thread, and GPU environments.

---

## 🚀 Usage

The benchmark script is **`benchmark.py`**.

### Command-line options

```bash
python benchmark.py --array <backend> --fft <fft_backend> [options]
```

#### Options

| Option | Description | Example |
|:--|:--|:--|
| `--array` | Array backend (`numpy` or `cupy`) | `--array numpy` |
| `--fft` | FFT backend (`scipy`, `fftw`, `cupy`) | `--fft fftw` |
| `--threads` | (For FFTW only) Number of threads | `--threads 4` |
| `--size` | Matrix size (HxH) | `--size 512` |
| `--niter` | Number of iterations | `--niter 1000` |
| `--profile` | Enable detailed profiling with `cProfile` | `--profile` |




```bash
# SciPy FFT (CPU)
python benchmark.py --array numpy --fft scipy --size 512 --niter 1000

# pyFFTW single-thread (CPU)
python benchmark.py --array numpy --fft fftw --threads 1 --size 512 --niter 1000

# pyFFTW multi-thread (CPU, 4 threads)
python benchmark.py --array numpy --fft fftw --threads 4 --size 1024 --niter 1000

# CuPy FFT (GPU)
python benchmark.py --array cupy --fft cupy --size 512 --niter 1000
```

## ⚙️ Benchmark Structure
Each iteration executes the following sequence:

### 1. FFT forward + inverse
```bash
b = fft2(a)
a_recon = ifft2(b)
```

### 2. Element-wise arithmetic
```bash
diff = abs(a - a_recon) ** 2
```

### 3. Reduction
```bash
s = sum(diff)
```

The benchmark measures total runtime and (when --profile is set) collects cumulative timings of each stage using cProfile.

## 📊 Results
Below are representative results for 1000 iterations across matrix sizes 256×256, 512×512, and 1024×1024.
Times are averages of 5 runs on the developer's laptop.

### ⚙️ Developer Environment

- **OS:** Windows 11 Home, ver. 24H2, OS build 26100.4652  
- **CPU:** Intel Core i7-14650K (16 cores, 24 threads)  
- **RAM:** 32 GB  
- **GPU:** NVIDIA RTX 4060 Laptop GPU (8 GB VRAM)  
- **NVIDIA Driver:** 576.02  
- **CUDA Toolkit:** 12.9  
- **Python:** 3.10.5 (venv)  
- **Libraries:**
  - NumPy: 2.2.6  
  - CuPy: 13.5.1  

**Note:** Results are device-dependent and may vary on different hardware or driver configurations.

### 🧪 SciPy FFT (CPU, PocketFFT)

| Metric | 256×256 | 512×512 | 1024×1024 |
|:--|--:|--:|--:|
| **Total time [s]** | 0.48 | 3.78 | 19.4 |
| **FFT share [%]** | 86 | 72 | 76 |
| **Element-wise share [%]** | 11 | 24 | 19 |
| **Reduction (sum) share [%]** | 3 | 2 | 1 |

**Conditions:**  
- FFT backend: `scipy.fft` (PocketFFT)  
- Array backend: `numpy` (CPU, single precision `complex64`)  
- Iterations: 1000  

**Observation:**   
- FFT dominates (~70–76 %), but the O(N) element-wise portion remains a significant ~20 %.  
- Reduction (`sum`) cost grows slowly (< 2 %).  


### 🧪 pyFFTW (threads=1)

| Metric | 256×256 | 512×512 | 1024×1024 |
|:--|--:|--:|--:|
| **Total time [s]** | 0.59 | 2.77 | 11.8 |
| **FFT share [%]** | 88 | 67 | 68 |
| **Element-wise share [%]** | 10 | 31 | 30 |
| **Reduction (sum) share [%]** | 2 | 2 | 2 |

**Conditions:**  
- FFT backend: `pyFFTW` (threads = 1, planner = `FFTW_MEASURE`)  
- Array backend: `numpy` (CPU, single precision `complex64`)  
- Iterations: 1000  

**Observation:**  
- Compared with `scipy.fft` (19.4 s at 1024²), pyFFTW is roughly **1.6× faster**.
- FFTW’s plan caching and float-precision kernels clearly pay off for larger arrays, making it the most efficient CPU-based option in this benchmark.


### 🧪 pyFFTW (threads=4)

| Metric | 256×256 | 512×512 | 1024×1024 |
|:--|--:|--:|--:|
| **Total time [s]** | 0.72 | 2.31 | 8.70 |
| **FFT share [%]** | 89 | 60 | 56 |
| **Element-wise share [%]** | 8 | 38 | 42 |
| **Reduction (sum) share [%]** | 3 | 2 | 2 |

**Conditions:**  
- FFT backend: `pyFFTW` (threads = 4, planner = `FFTW_MEASURE`)  
- Array backend: `numpy` (CPU, single precision `complex64`)  
- Iterations: 1000  

**Observation:**  
- At 1024², total runtime ≈ **8.7 s**, giving **1.36× overall speed-up** over the 1-thread case (11.8 s).  
- FFT time reduces to ~4.8 s (≈ 56 %), but the **O(N) element-wise step (3.6 s, 42 %)** now dominates runtime.  



### 🧪 CuPy FFT (GPU, cuFFT)

| Metric | 256×256 | 512×512 | 1024×1024 |
|:--|--:|--:|--:|
| **Total time [s]** | 0.21 | 0.22 | 0.51 |
| **FFT share [%]** | 61 | 62 | 26 |
| **Element-wise share [%]** | 22 | 20 | 58 |
| **Reduction (sum) share [%]** | 16 | 15 | 7 |

**Conditions:**  
- FFT backend: `cupy.fft` (cuFFT via CuPy)  
- Array backend: `cupy` (GPU, single precision `complex64`)  
- Device: NVIDIA RTX 4060 Laptop GPU (CUDA 12.9, driver 576.02)  
- Iterations: 1000  

**Observation:**
- At 1024², FFT time increases slightly (0.13 s) but remains just **25 % of total runtime**, while the element-wise operation now dominates (**~60 %**) due to global memory traffic.  
- These results clearly demonstrate that once FFT kernels reach full throughput, **the GPU performance ceiling is set by memory bandwidth**, not compute.  
- Compared to the optimized CPU FFTW (4 threads, 8.7 s at 1024²), the GPU achieves an overall **≈17× speedup**, illustrating how GPU-accelerated FFTs  
  shift the bottleneck entirely to O(N) memory-bound stages.

---

### 💡 Practical note

These benchmarks show that **FFT itself is rarely the bottleneck** in GPU computing. Instead, **memory-bound O(N)** stages — such as element-wise arithmetic or reductions — dominate total runtime once FFT throughput saturates.

In practice, this limitation can be mitigated through **kernel fusion**, which reduces redundant global-memory traffic by combining multiple operations into a single CUDA kernel.  
For FFT workloads, NVIDIA’s [**cuFFT Callback API**](https://nw.tsuda.ac.jp/lec/cuda/doc_v9_0/pdf/CUFFT_Library.pdf) provides a built-in mechanism for this purpose.