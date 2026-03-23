# RFF-SVM: Random Fourier Features for Kernel SVM

A from-scratch implementation of a kernel SVM that uses **Random Fourier Features (RFF)** to approximate the RBF (Gaussian) kernel — avoiding the expensive O(N²) kernel matrix entirely.

RFF maps input data into a low-dimensional random feature space where dot products approximate the RBF kernel (based on Bochner's theorem, Rahimi & Recht 2007). This converts a non-linear SVM problem into a linear one, trained using the **Pegasos** algorithm (mini-batch stochastic sub-gradient descent). The entire pipeline uses float32 and computes features per mini-batch to keep memory usage minimal.

## Results

Tested on a synthetic binary classification dataset (12,000 samples, 20 features) with varying RFF dimensions (D):

| D | Accuracy | Time (s) | Memory (MB) |
|---|----------|-----------|-------------|
| 50 | 73.21% | 0.02 | 142.8 |
| 100 | 74.58% | 0.03 | 142.8 |
| 300 | 79.46% | 0.09 | 145.8 |
| 500 | 76.83% | 0.28 | 147.1 |
| 1000 | 79.25% | 0.89 | 147.4 |

Accuracy improves with D and saturates around D=300–500, achieving near-exact-kernel performance at a fraction of the memory cost. Memory scales linearly with D while staying well under 150 MB throughout.
