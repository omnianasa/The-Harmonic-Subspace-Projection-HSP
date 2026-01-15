# Algorithm Developing To reach the big one

### Basic Idea 

we need to make a new approach of AI NN that encourage for general learning in lower cost, not black box, and more accurate. How can we make a new design of NN and the one we have is good?

###### Does recent NN Behave well? 

Not really, but it works
current networks make great performance but suffer from high computation(GPU), catastrophic forgetting problem and black box representations

### How to develop?

We rely on 2 things:
1- Ray tracing concept to apply ray base neurons 
2- biologically, how we see? 
high resolution is processed at the center of gaze (fovea), low resolution is peripheral region. The brain works on and integrate them and increase the resolution (similar as CNN)
apply ray tracing like we have studied in computer graphic but as the base idea in the NN it will be the gradient and acts like a ray that travels between nodes. Bfocus in big and then become blur a edges that the brain make it more stronger so we will reach a similar design using the ray tracing which will terminate the black box of NN 

### What exact problem am I trying to solve?

I need to rethink in the digital neuron from the ground up. Instead of just following the same old path, I need to build a new architecture that changes how data is represented and how networks learn. My goal is to replace slow, expensive, and mysterious systems with a new model that is fast, clear, and easy to scale and make the next revolution in the field of AI.
 - how to correctly represent data
 - how to build the neuron and the network 
 - how it is scalable for most applications 
 - be more clear without the blackbox 

### Which limitation is primary: computation cost, black box, catastrophic forgetting?

The plan is to apply as most as i can but will be in different phases starting with: black box, forgetting, computation
*why*:
1- black box: making a new algorithm that solves the black box may open new chances to get the actual black box problem
2- forgetting: i thought in making it the first periority but if you know the path (the right one) we can get how to return it back and have a virtual memory or in-memory inside the neuron to update it.
3- computation: even with new algorithm and building a (network) that will need big computations for taking the weight and parameters based on.   

### Why current neural network methods can not solve this problem with some modifications?

from little search i have made until now :
For Catastrophic Forgetting: While early approaches struggled with this "flaw" (Zohuri & Moghaddam, 2020), modern research into neuromorphic computing offers a more robust solution. Tan et al. (2020) argue that the event-driven nature of SNNs, combined with Spike-Timing Dependent Plasticity (STDP), mimics the synaptic consolidation found in the human brain
For Black Box Problem: In traditional deep learning, decisions are built by millions of weights, making it almost impossible to trace the logic of a specific output. However, the development of Spiking Neural Networks (SNNs) and the NeuCube architecture has shifted the paradigm from abstract matrices to spatial transparency.

BUT ALL ALGORITHMS ARE NOT STILL COMPLETE AND LOSE THE SOLUTION ACC OR NOT SCALABLE 

### About the biological phase 

think that i need to develop SNN which improves the biological phase in very good details. but the algorithm needs to be developed in ;
- computations and efficiency
- New Neuron Model
- Catastrophic forgetting co-design

## More inside the idea

so we need to make a neural network less computational, do not forget, better performance. Based on the  ray tracing that inspired me to apply it we can replace the numbers it all based on the paths not just the path the weigt got to as we discussed but replacing the full numbers to ways and paths.. just when we add new tasks the old paths will not be removed which reduce the percentage of forgetting. 

## How does foveated vision translate into a computational methods?
we do not see in the same resolution. the center of foveated is the most resolution accuracy and it reduces on periphery. the foveated is based on the attention and the brain is the master to integrate all that together . 
SO THAT: not all input dimensions deserve equal precision or equal learning effort. So instead of uniform tensors, use multi resolution, attention guided representations.


The data will be represented in 3 levels general, special, very special. The foveated is where the network focuses on based on what?
- new thing
- a mistake 
- lots of rays

FOVETAED -> low and strong rays
PERIPHERY -> high and weak
The network learn from what it focuses on

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





## QUESTIONS TO BE ASKED

- What exact problem am I trying to solve? **ok**
- Which limitation is primary: computation cost, black box, catastrophic forgetting? **ok**
- Why current neural network methods can not solve this problem with some modifications? **ok**
- Which biological principles am I using? **ok**
- Which biological details am I ignoring, and why? **ok**
- Am I borrowing structure from biology or only metaphor? **ok**
- What does biological vision do better than current neural networks? **ok**
- How does foveated vision translate into a computational methods? **ok**
- What does success look like in measurable terms? **ok**
- Which metrics will be used to check improvement? **ok**
- What is the fundamental unit for computation in this model? ray **ok**
- What replaces standard back propagation? **ok**
- How will information propagate through the system? active rays only based on the fovea **ok**
- What defines the start of a learning signal (ray)? attention **ok**
- What defines its direction?
- What defines its termination?
- Is the process deterministic or probabilistic? both
- Can a single decision be traced end-to-end?
- Are learning paths visible? yes fully **ok**
- What is the smallest experiment that tests the core hypothesis?
- What dataset is sufficient?
- What outcome would make the hypothesis successful?
- How results could be compared to a baseline?