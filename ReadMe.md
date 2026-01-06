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


## QUESTIONS TO BE ASKED

- What exact problem am I trying to solve?
- Which limitation is primary: computation cost, black box, catastrophic forgetting?
- Why current neural network methods can not solve this problem with some modifications?
- Which biological principles am I using?
- Which biological details am I ignoring, and why?
- Am I borrowing structure from biology or only metaphor?
- What does biological vision do better than current neural networks?
- How does foveated vision translate into a computational methods?
- What does success look like in measurable terms?
- Which metrics will be used to check improvement?
- What is the fundamental unit for computation in this model?
- What replaces standard back propagation?
- How will information propagate through the system?
- What defines the start of a learning signal (ray)?
- What defines its direction?
- What defines its termination?
- Is the process deterministic or probabilistic?
- Can a single decision be traced end-to-end?
- Are learning paths visible?
- What is the smallest experiment that tests the core hypothesis?
- What dataset is sufficient?
- What outcome would make the hypothesis successful?
- How results could be compared to a baseline?