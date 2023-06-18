# Week 6
## Question 1
Interpretability work on artificial neural networks is closely related to interpretability work on biological neural networks (aka brains). Describe two ways in which the former is easier than the latter, and two ways in which it’s harder.
### My original answer
Working on ANNs is easier because the models can be saved and shared with perfect fidelity, and thus results are eminently repeatable. ANNs do not require the complex biological infrastructure required for cell culture, and can be probed just by reading some floating point numbers out of memory: no special instruments required.

Working on ANNs is harder mostly because it is relatively unexplored. Neuroscience has the advantage of well established standards for research and familiarity with the relevant motifs. Also, ANNs become more interesting to study as they increase in size, but may become prohibitively expensive to train at those sizes.
### Answer co-written with an LLM
Working on ANNs is easier compared to brains in the following ways:
1. Perfect fidelity and repeatability: ANNs can be saved and shared with perfect fidelity, enabling results to be replicated accurately. This makes experimentation and analysis more reliable and repeatable compared to studying biological neural networks, which require complex infrastructure for cell culture and face challenges in reproducibility.
1. Simplicity of probing: ANNs can be probed simply by reading floating-point numbers from memory, without the need for specialized instruments. This ease of access facilitates analysis and interpretation of the network's inner workings.

However, working on ANNs also presents challenges:
1. Relative novelty: The field of interpretability in ANNs is relatively unexplored compared to neuroscience. Neuroscience benefits from well-established research standards and a deeper understanding of the relevant motifs, while interpretability in ANNs is still an evolving area.
1. Scalability and cost: Larger ANNs tend to be more interesting to study, but training them at scale can become prohibitively expensive. This poses challenges in analyzing and interpreting the behavior and inner workings of complex and computationally intensive networks.

In summary, while ANNs offer advantages such as perfect fidelity and simplicity of probing, challenges lie in the relatively unexplored nature of interpretability research and the scalability and cost of studying larger networks.
## Question 2
What are the most important disanalogies between POMDPs and the real world?
### My original answer
POMDPs deal with a discrete state space and discrete actions, while the real world is continuous. POMDPs also cleanly separate the agent from the environment, and the environment from the reward function. A truly embodied agent will have all kinds of back channels to change itself and its reward function by taking actions in the environment.
### Answer co-written with an LLM
The most important disanalogies between POMDPs and the real world are as follows:
1. Discrete vs. Continuous: POMDPs typically deal with discrete state spaces and discrete actions, whereas the real world operates in continuous spaces. The discrete nature of POMDPs oversimplifies the continuous and nuanced aspects of the real-world environment.
1. Simplified Model Assumptions: POMDPs cleanly separate the agent, the environment, and the reward function, making it easier to reason about their interactions. However, in the real world, an embodied agent exists within a complex and interconnected system, with numerous feedback loops and intricate dependencies.

In summary, POMDPs differ from the real world in terms of the discrete nature of state and action spaces, simplified model assumptions, and the limited feedback channels compared to the rich sensory capabilities and complex interconnections present in real-world environments.
## Discussion Prompts
### Prompt 3
Were you surprised by the results and claims in Zoom In? Do you believe the Circuits hypothesis? If true, what are its most important implications?
#### Answer 3
I was a little surprised. I think the Circuits hypothesis is likely one of those things that is true in practice, because it is an effective solution relatively easy to find via search processes. But it may not be true in principle. It’s probably possible to create obfuscated cognition of some kind, and this may rarely be found first by optimization.