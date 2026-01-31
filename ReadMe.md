# Algorithm Developing To reach the big one

## Abstract
Standard neural networks suffer from **Catastrophic Forgetting** because every weight is updated for every task, leading to the overwriting of prior knowledge. This project introduces **Ray Allocation**, an architecture that treats neural capacity as a finite set of "Rays" (functional pathways). By dynamically routing information through multi-scale resolutions and freezing rays post-acquisition, we achieve task isolation and memory preservation without a fixed spatial fovea.

---

## Stage 1: Mathematical Foundation

### 1. Multi-Scale Representation
Inputs are decomposed into a hierarchy of feature resolutions to separate global topology from local precision:
$$X \longrightarrow \{ \Phi^{(0)}, \Phi^{(1)}, \dots, \Phi^{(K)} \}$$
* **$\Phi^{(0)}$**: Coarse, low-frequency global structure.
* **$\Phi^{(K)}$**: Fine-grained, high-frequency local details.

### 2. Adaptive Ray Allocation
A "Ray" $r \in \mathcal{R}$ represents a specific learnable pathway. Selection is governed by a **Score Function** ($S$) that prioritizes novelty and error:
$$S = \text{Activation} \times \text{Error} \times e^{-\beta \cdot \text{UsageCount}}$$
* **UsageCount**: Tracks how many tasks have utilized a specific ray.
* **$\beta$**: A penalty factor that pushes the model to use "virgin" capacity for new tasks.

### 3. Gradient Gating
To prevent forgetting, updates are restricted to the active foveal-ray set $\mathcal{F}_t$:
$$\Delta w = 
\begin{cases} 
\eta \cdot \nabla L & \text{if } w \in \text{Active Ray } r \\
0 & \text{if } w \in \text{Protected Ray } r
\end{cases}$$

---

## Stage 2: Architecture & Development Stages

### Phase I: Structural Sparsity
We replace dense layers with a **Ray Bank**. Instead of a single forward pass through all neurons, the network routes the input through three scales: **Global (Coarse)**, **Context (Medium)**, and **Detail (Fine)**. Each scale consists of independent "Ray" units.



### Phase II: Dynamic Routing
Rays are not static. The model computes a selection score based on prediction error. If the current active rays cannot represent the input (high error), the model recruits "inactive rays" from the bank to expand its knowledge.

### Phase III: Memory Preservation (The Ray Shield)
When Task $A$ is complete, the rays used for its success are "shielded." For Task $B$:
* **Frozen Ray Memory**: Activations from Task $A$ rays are added as a residual input to provide context.
* **Task-Specific Heads**: Separate output layers map the combined activations to task-specific labels, preventing "readout interference."

---

## Stage 3: Scale-Dependent Penalty
Global features (coarse scale) can often be shared across tasks (e.g., both "car" and "truck" need to know what a wheel looks like). However, fine-grained details must be unique. We implement a scale-dependent penalty:
$$\beta_{\text{scale}} = \beta \cdot (\text{scale\_index} + 1)^2$$
This allows the **Coarse Rays** to be communal, while **Fine Rays** become highly specialized experts for specific tasks.

---

## Stage 4: Evaluation & Success Metrics

We measure the health of the system through three primary lenses:

### 1. Performance Metrics
* **Accuracy (Top-1/5):** Tracking performance on new tasks.
* **Backward Transfer:** Measuring how much performance on Task 1 drops after training Task 5.
* **Computational Efficiency:** Measured in **FLOPS** and **Active Parameter Count** (Sparsity).

### 2. Structural Metrics
* **Ray Overlap ($\mathcal{O}$):** The fraction of shared rays between tasks.
    $$\mathcal{O}_{AB} = \frac{|\mathcal{R}_A \cap \mathcal{R}_B|}{|\mathcal{R}_A|}$$
* **Ray Saturation:** The rate at which the "Ray Bank" is being depleted.



### 3. Robustness
Resistance to noise and the model's ability to maintain high precision even when partial scales are suppressed.

---

## Conclusion
This Ray-based approach allows a single model to grow continuously. By balancing the sharing of coarse global features with the strict isolation of fine-scale rays, we mimic the efficiency of modular biological systems while maintaining the mathematical rigor of modern gradient descent.