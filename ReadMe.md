# Algorithm Developing To reach the big one

## What is RayNN?
RayNN is a new type of neural network designed to:
- Learn general features efficiently.
- Use less computation.
- Be interpretable (you can understand how it works).
- Remember knowledge across tasks (avoiding forgetting).

## Why Do We Need It?
Current neural networks:
- Need a lot of GPU and computation.
- Forget previously learned tasks easily.
- Are like "black boxes": we cannot easily see how they make decisions.

## How RayNN Works
RayNN is inspired by:
1. **Ray tracing in computation**: Neuron connections act like rays that carry information and learning signals.
2. **Biological vision**: Like the eye, RayNN focuses on important parts of input with high detail (fovea) and less important parts with low detail (periphery).

### Key Problems RayNN Solves
- Efficiently represents data.
- Makes the network more interpretable.
- Scales well for multi-task learning.
- Reduces forgetting old tasks.

### Main Goals
1. Make the network interpretable.
2. Reduce forgetting of old tasks (using memory-based fovea).
3. Reduce computation cost (later development).

## Why Current Networks Fail
- They forget old tasks because weights get overwritten.
- They are hard to interpret because of millions of connections.
- Some neuromorphic approaches improve this, but not completely.

## Biological Inspiration
- Vision works at multiple resolutions: detailed at the center, coarse in the periphery.
- Attention focuses learning on novel or error-prone regions.
- Rays in neurons are like spikes traveling along neural paths.

## How Foveated Vision Works in RayNN
- Not all input is equally important.
- Multi-scale, attention-guided representations are used instead of uniform data.
- Three levels of focus:
  - Global (coarse)
  - Special (medium)
  - Fovea (very detailed)
- Rays in the fovea are strong and precise; rays in the periphery are weaker.

### Attention-Guided Ray Propagation
- Rays carry information through the network like signals in the brain.
- Attention focuses rays on important regions of the input.
- Weak rays still exist in less important areas but are less precise.
- This improves efficiency and interpretability.

### Memory-Based Fovea for Continual Learning
- The fovea has a memory of important features from previous tasks.
- Helps the network remember old tasks while learning new ones.
- Reduces catastrophic forgetting by selectively updating connections

# Mathematical explanation

## 1. Multi-scale Input Representation
input $X$ is structured into a hierarchy of resolutions rather than a single dense vector:

$$X \longrightarrow \{ X^{(0)}, X^{(1)}, \dots, X^{(K)} \}$$

* $X^{(0)}$: Coarse, global structure.
* $X^{(K)}$: Fine, local details.

## 2. Fovea Selection (Dynamic Attention)

$$\mathcal{F}_t = \arg \max_{\Omega \subseteq X} S(X, \Omega)$$

multiple signals to guide focus:
$$S(X, \Omega) = \alpha_1 \text{Uncertainty}(\Omega) + \alpha_2 \text{Novelty}(\Omega) + \alpha_3 \text{PredictionError}(\Omega) + \alpha_4 \text{RayDensity}(\Omega)$$

## 3. Ray Allocation Across Scales
Ray intensity $\alpha_k$ for each scale $k$ follows an exponential decay to balance the global context and local precision:

$$\alpha_k = \alpha_0 \cdot e^{-\lambda k}, \quad k=0,1,\dots,K$$

Where:
* $\alpha_0$: Base ray intensity.
* $\lambda$: Scale decay factor.
* Each ray $r \in \mathcal{R}_k$ carries a feature representation $\phi(r)$.

## 4. Learning Updates Restricted to Fovea
To prevent overwriting existing knowledge, weight updates are strictly localized to the current fovea:

$$\Delta \phi(r) = 
\begin{cases} 
\text{learning\_update}(\phi(r)) & \text{if } r \in \mathcal{F}_t \\
0 & \text{if } r \notin \mathcal{F}_t
\end{cases}$$

## 5. Integration of Rays into Output
The final output $y$ integrates features across all scales:

$$y = f\Bigg(\sum_{k=0}^{K} \sum_{r \in \mathcal{R}_k} \alpha_k \cdot \phi(r) \Bigg)$$

## 6. Task Separation for Continual Learning
spatial and structural task isolation:
* **Unique Foveae:** $\mathcal{F}_A \neq \mathcal{F}_B$
* **Disjoint Rays:** $\mathcal{R}_A \cap \mathcal{R}_B \approx \emptyset$

## What does success look like in measurable terms?
F1 core, Precision, Top1-5 accuracy, FLOPS, GPU/CPU usage, Number of parameters(memory)
Ray Coverage Map, Fovea overlap, active ray, sparsity (processed/possible)
Robustness