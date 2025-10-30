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

### 🧪 SciPy FFT (CPU, PocketFFT) – Performance Summary

| Metric | 256×256 | 512×512 | 1024×1024 |
|:--|--:|--:|--:|
| **Total time [s]** | 1.8 | 8.4 | 42.8 |
| **FFT share [%]** | 70 | 72 | 77 |
| **Element-wise share [%]** | 25 | 22 | 17 |
| **Reduction (sum) share [%]** | 1 | 1 | 1 |

**Conditions:**  
- Backend: `scipy.fft` (PocketFFT)  
- Array backend: `numpy` (CPU)  
- Iterations: 1000  
- Each value is based on averaged or representative profiling runs.  

**Observation:**  
- FFT dominates roughly 70–77% of runtime across sizes,  
  but element-wise operations still account for 15–25%, even at 1024².  
- Reduction remains consistently negligible (<2%).  
- The relative proportions change slowly with size — FFT scales as expected (O(N log N)),  
  yet memory-bound O(N) element-wise work remains significant.


### 🧪 pyFFTW (threads=1, planner=FFTW_MEASURE) – Performance Summary

| Metric | 256×256 | 512×512 | 1024×1024 |
|:--|--:|--:|--:|
| **Total time [s]** | 1.25 | 4.90 | 22.5 |
| **FFT share [%]** | 59 | 62 | 62 |
| **Element-wise share [%]** | 37 | 36 | 36 |
| **Reduction (sum) share [%]** | 2 | 2 | 2 |

**Conditions:**  
- FFT backend: `pyFFTW` (threads = 1, planner = `FFTW_MEASURE`)  
- Array backend: `numpy` (CPU)  
- Iterations: 1000  
- Each value based on representative profiling runs.

**Observation:**  
- FFTW consistently outperforms `scipy.fft` (PocketFFT) by roughly **1.7–2×** across all tested sizes.  
- The **FFT share remains around 60 %**, indicating that even with a highly optimized FFT engine,  
  **O(N)** element-wise operations still account for about one-third of total runtime.  
- Reduction (sum) operations stay negligible (< 2 %).  
- The scaling behavior (≈ ×4–5 runtime increase for each doubling of size)  
  follows the expected *O(N log N)* trend of FFT workloads.  
- Because `planner_effort = FFTW_MEASURE` was used, plan creation overhead was excluded from timing,  
  and cached plans were reused across iterations.


### 🧪 pyFFTW (threads=4, planner=FFTW_MEASURE) – Performance Summary

| Metric | 256×256 | 512×512 | 1024×1024 |
|:--|--:|--:|--:|
| **Total time [s]** | 1.26 | 3.73 | 15.9 |
| **FFT share [%]** | 63 | 53 | 50 |
| **Element-wise share [%]** | 35 | 44 | 47 |
| **Reduction (sum) share [%]** | 2 | 3 | 3 |

**Conditions:**  
- FFT backend: `pyFFTW` (threads = 4, planner = `FFTW_MEASURE`)  
- Array backend: `numpy` (CPU)  
- Iterations: 1000  
- Each value based on representative profiling runs.

**Observation:**  
- At 1024², total runtime ≈ **15.9 s**, ~**1.4× faster** than the single-threaded case (22.5 s).  
- FFT phase consumes ~8.0 s (≈ 50 %), while element-wise work already reaches **7.5 s (≈ 47 %)**—nearly equal contributions.  
- The small residual (≈ 0.45 s) comes from reduction.  
- This run demonstrates the classic **Amdahl’s law saturation**: even with efficient multi-threaded FFTs, the non-FFT portion (memory-bound O(N) operations) becomes the limiting factor for overall performance.


### 🧪 CuPy FFT (GPU, cuFFT) – Performance Summary

| Metric | 256×256 | 512×512 | 1024×1024 |
|:--|--:|--:|--:|
| **Total time [s]** | 0.28 | 0.76 | 3.23 |
| **FFT share [%]** | 51 | 50 | 47 |
| **Element-wise share [%]** | 19 | 17 | 17 |
| **Reduction (sum) share [%]** | 21 | 25 | 26 |

**Conditions:**  
- FFT backend: `cupy.fft` (cuFFT via CuPy)  
- Array backend: `cupy` (GPU)  
- Device: NVIDIA GPU (cuFFT backend)  
- Iterations: 1000  
- Profiling via `cProfile` with synchronization after loop.

**Observation:**  
- Total runtime increases to **3.23 s** at 1024², still **5–7× faster** than optimized CPU FFT (FFTW, 4 threads).  
- FFT: 1.50 s (≈ 47 %), Reduction: 0.85 s (≈ 26 %), Element-wise: 0.53 s (≈ 17 %).  
- Despite the larger workload, **reduction (`cupy.sum`) remains a major bottleneck**—its cost grows almost linearly with data size, reflecting the memory-bandwidth limit of GPU global reads.  
- FFT shows good scaling with cuFFT, but further acceleration yields diminishing returns because the reduction and element-wise kernels increasingly dominate the overall runtime.  
- This illustrates the **complete inversion of bottlenecks** compared with CPU results:  
  while CPU time was FFT-dominated, on GPU the non-FFT components (especially reduction) govern total performance.

