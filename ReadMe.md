# Harmonic Subspace Projection (HSP)

> Just personal experiment

## I. Overview
The **Harmonic Subspace Projection (HSP)** is a new way to classify data without the guessing game of traditional AI. Instead of using standard deep learning, HSP treats data like radio signals. By using geometry and physics, it maps information onto resonant manifolds. 

In tests with the MNIST dataset, HSP achieved **94.70% accuracy** and can process **32,232 images per second** making it incredibly fast and mathematically transparent.


## II. The Problem with "Black Box" AI

Most modern AI models are "Black Boxes." They learn through trial and error (backpropagation), making it hard to explain exactly why a model chose a specific answer. They also require massive amounts of computing power to train.

HSP tries to solve this by treating classification as a signal processing task. It identifies a unique resonant space for every category. By seeing where a new piece of data vibrates the loudest, HSP finds the answer instantly without needing a single training iteration.

---

## III. How It Works

### 1. Creating the Ray
We take input data (like an image) and turn it into a high frequency signal called a **Ray**. We use a fixed mathematical projection to make sure every image is represented as a unique oscillating signature.

### 2. Building the Subspaces
Instead of drawing lines between categories, HSP builds a geometric subspace (like a room) for each one. We look at the training data for a specific class (like the number 5) and extract the most important patterns to form a **Class Basis**. This is the resonant subspace where that category lives.

### 3. Finding the Resonance
To check a new image, we bounce its Ray off every category's room at the same time. We measure the **Resonance Energy** (the strength of the echo). The category with the strongest vibration is the winner.

## IV. Experimental Results & Comparative Analysis

### 4.1 Performance Benchmarking (MNIST)
*Sample size: 20k Training / 5k Testing*


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

---

## V. Diagnostic Analysis (White-Box Benefits)

The primary advantage of HSP is treating classification as a resonance problem, we can generate a "mathematical receipt" for every decision, allowing for real-time auditing of model confidence and manifold health.

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
<tr>
<th width="50%">Phase 1: Latent Manifold Separation</th>
<th width="50%">Phase 2: Signal Quality & Clarity</th>
</tr>
<tr>
<td align="center">
<img src="classification/results/manifold.png" width="90%" />



<sub><b>Fig 1:</b> t-SNE reduction proving that Harmonic Rays create naturally separable clusters before final basis extraction.</sub>
</td>
<td align="center">
<img src="classification/results/snr.png" width="90%" />



<sub><b>Fig 2:</b> Distribution of SNR; distinct peaks demonstrate a high margin of safety between resonant and non-resonant classes.</sub>
</td>
</tr>
</table>

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
<tr>
<th width="100%">Phase 3: Decision Logic (Resonance Spectrum)</th>
</tr>
<tr>
<td align="center">
<img src="classification/results/dig2.png" width="60%" />



<sub><b>Fig 3:</b> The Resonance Spectrum for a sample input. Decision-making is transparent, showing exactly how much energy was captured by each class subspace.</sub>
</td>
</tr>
</table>
---

## VI. Conclusion

HSP is a "(informal) lean and mean" alternative to heavy AI, but it is still evolving. 

* **Current Weakness:** It uses a single layer to look at data. Unlike deep neural networks that see details and then shapes, HSP looks at everything at once. This can make it struggle with very complex or messy images.

* **Next Steps:** I am working on making the system adaptive so it can focus on important details automatically, sharpening its accuracy without losing its signature speed and transparency.

---