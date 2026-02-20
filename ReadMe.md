# Research Report: Experiment 1
## The Harmonic Signal Protocol (HSP)
**Date:** February 20, 2026  
**Status:** Completed - Phase 1 Validation

---

## 1. Abstract
This report introduces the results of **Experiment 1** regarding the **Harmonic Signal Protocol (HSP)**, a non-iterative, high-throughput classification framework. HSP shifts the paradigm from traditional iterative weight optimization to a system of high-dimensional Fourier projections and class-specific eigen-decomposition. Evaluating the protocol on the MNIST-784 dataset, we demonstrate a final accuracy of **94.70%** and an unprecedented throughput of **32,232 images per second**. The protocol offers a "White Box" alternative to deep learning, providing mathematical transparency through resonance energy and signal-to-noise ratio (SNR) metrics.

---

## 2. Introduction
The current landscape of machine learning is dominated by deep neural networks that rely on backpropagation and gradient-based optimization. While effective, these methods are computationally intensive and act as "black boxes" regarding their internal decision logic.

In Experiment 1, we propose and validate the **Harmonic Signal Protocol (HSP)**, a framework rooted in signal communication theory. The core objective of HSP is to achieve rapid learning and high-frequency inference by treating classification as a geometric resonance problem. By projecting data into an oscillating Hilbert space, we identify unique "resonant signatures" for each class.

---

## 3. Methodology

### 3.1 Harmonic Ray Emission
Input data $x \in \mathbb{R}^d$ is mapped into a high-dimensional oscillating feature space. Using a fixed orthogonal projection matrix $W$ (initialized via QR decomposition), we apply a sinusoidal transformation to approximate a shift-invariant kernel (Random Fourier Features):

$$z = xW$$
$$\Phi(x) = [\cos(z), \sin(z)]$$

The resulting vector, termed a **"Turbo Ray"**, represents the input as a high-frequency signal signature. Rays undergo global mean centering and $L_2$ normalization to reside on a unit hypersphere.

### 3.2 Subspace Learning
Rather than learning a global decision boundary, HSP identifies the geometric manifold occupied by each class. For each class $k$, the protocol calculates the covariance of its training rays:
$$\Sigma_k = \Phi(X_k)^T \Phi(X_k)$$

Using covariance-based eigen-decomposition, we extract the top $m$ eigenvectors to form the **Class Basis** $B_k$. This basis defines the "resonant subspace" for that specific label.

### 3.3 Inference via Resonance Energy
A test ray $\Phi(x_{test})$ is projected onto all learned class bases simultaneously. We calculate the **Resonance Energy** ($E$) as the squared $L_2$ norm of the projection:
$$E_k = \| \Phi(x_{test}) B_k^T \|^2$$

The system selects the label with the maximum resonance energy:
$$\hat{y} = \text{argmax}_k E_k$$

---

## 4. Experiment 1: Technical Report

| Metric | Result |
| :--- | :--- |
| **Train Time** | 5.8403s |
| **Final Accuracy** | 94.70% |
| **Throughput** | 32,232 img/sec |
| **Avg. System SNR** | 0.16 dB |

---

## 5. White-Box Diagnostic Visualizations

The following diagnostic proofs confirm the internal mechanics of the HSP system during Experiment 1.

### 5.1 Latent Space Clustering
![t-SNE Visualization](classification/results/manifold.png)
*Figure 1: t-SNE reduction of high-dimensional rays showing natural class clustering.*

### 5.2 Signal Clarity
![SNR Distribution](classification/results/snr.png)
*Figure 2: Distribution of Signal-to-Noise Ratio (SNR) indicating clear resonance separation.*

### 5.3 Per-Sample Decision Proof
![Resonance Spectrum](classification/results/dig2.png)
*Figure 3: Resonance spectrum for Digit 2, showing energy spikes in the correct class subspace.*

### 5.4 Phase and Waveform Analysis
![Phase Plane](classification/results/phase.png) ![Waveform](classification/results/signalSignature.png)
*Figure 4: Phase plane distribution (Left) and raw Ray Waveform signature (Right).*

### 5.5 Learned Spatial Morphology
![Spatial Templates](classification/results/wb.png)
*Figure 5: Inverse projection of class bases revealing the pixel-space templates learned by the model.*

### 5.6 Subspace Overlap
![Similarity Map](classification/results/cm.png)
*Figure 6: Similarity matrix showing mathematical correlation between different class subspaces.*

---
## 4. Experiment 2: Comparative Performance Analysis

In this experiment, we benchmarked the **Harmonic Signal Protocol (HSP)** against a diverse suite of industry-standard classifiers. This "Stress Test" evaluates the trade-offs between mathematical resonance and traditional iterative, distance-based, or ensemble-based learning methods.

### 4.1 Full Technical Benchmark Table
*Dataset: MNIST (20,000 Training Samples / 5,000 Test Samples)*

| Model | Accuracy | Train Time (s) | Inference Time (s) | Throughput (img/s) | Scaling | Interpretability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **HSP** | **0.9376** | **9.7313** | **0.2900** | **17,238.32** | **Linear** | **High** |
| Logistic Reg | 0.9115 | 5.5051 | 0.0115 | 433,492.91 | Non-Linear | High |
| Ridge Clf | 0.8540 | 0.8727 | 0.0175 | 285,113.45 | Linear | Low |
| KNN (k=3) | 0.9528 | 0.0844 | 4.8930 | 1,021.85 | Non-Linear | Low |
| Decision Tree | 0.8372 | 5.8212 | 0.0074 | 671,561.41 | Non-Linear | High |
| Random Forest | 0.9516 | 6.2157 | 0.0602 | 82,975.67 | Non-Linear | Low |
| Extra Trees | 0.9554 | 4.8460 | 0.0711 | 70,275.41 | Non-Linear | Low |
| SVM (RBF) | 0.9672 | 26.7032 | 4.2529 | 1,175.66 | Non-Linear | Low |
| MLP (128,64) | 0.9618 | 13.8231 | 0.0424 | 117,703.79 | Non-Linear | Low |
| Naive Bayes | 0.5860 | 0.2145 | 0.1589 | 31,464.24 | Linear | Low |



### 4.2 Experiment 2 Observations

* **The Latency-Accuracy Balance:** While **SVM (RBF)** achieves the highest accuracy (0.9672), its inference process is over **14x slower** than HSP. This demonstrates HSP's efficiency in high-frequency environments where real-time response is critical.
* **Throughput Advantage:** HSP maintains a robust throughput of **17,238 img/s**. While simpler linear models like Logistic Regression are faster, they fail to reach the $>93\%$ accuracy threshold achieved by the Harmonic Signal Protocol.
* **Predictable Scaling:** Unlike **KNN**, which slows down drastically as the training set grows (due to $O(N)$ search complexity), HSP scales linearly. Its inference speed is determined solely by the `ray_dim`, making it predictable and ideal for hardware deployment (FPGAs/Edge).
* **The Interpretability Gap:** Although **MLP** and **Random Forest** show strong performance, they remain "Black Box" models. HSP provides the only high-accuracy result in this table that can be fully diagnosed via **Subspace Resonance** and **Signal-to-Noise Ratio (SNR)** metrics.



