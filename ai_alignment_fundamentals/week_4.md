# Week 4
## Question 1
Imagine using reward modelling, as described in the second reading from this week, to train an AI to perform a complex task like building a castle in Minecraft. What sort of problems would you encounter?
### My original answer
Primarily, I’d expect that building a castle would run into issues because it is a task which is quite long, with a lot of complex subtasks. It would be difficult for a human to evaluate which of the two snapshots is better without the context surrounding them. For example, it might be useful to mine stone to build the walls with, but not if there are chests nearby with lots of stone saved from earlier mining.

To a lesser extent, a castle is a pretty vague concept, much more so than a backflip, with lots of ways to create one, with lots of steps that can be performed in any order. I think there’s a chance the reward predictor wouldn’t be able to learn it well.

Certain behaviors which undo each other can be good in subtly different circumstances, and with the lack of context in the samples the human is to judge, the RL algorithm could end up incentivized to loop on these pairs of actions. For example, mining stone and placing stone.
### Answer co-written with an LLM
When using reward modeling to train an AI to build a castle in Minecraft, several problems can arise:

1. Complex and lengthy task: Building a castle involves multiple complex subtasks, which can make it challenging to evaluate progress or determine the quality of intermediate snapshots without proper context. For example, it might be difficult to judge the significance of mining stone for building walls if there are nearby chests with ample stone already collected.
1. Vagueness and variability: The concept of a castle is relatively vague, with multiple valid ways to construct one and numerous steps that can be performed in different orders. This variability may pose difficulties for the reward predictor to accurately capture the desired criteria for castle construction.
1. Conflicting behaviors: Certain actions can be beneficial in specific circumstances but counterproductive in others. With limited context available in the samples for human evaluation, the RL algorithm may inadvertently incentivize behavior loops that involve conflicting actions, such as mining stone and immediately placing it back.

In summary, the complexity and length of the task, the vagueness and variability of castle construction, and the potential for conflicting behaviors pose significant challenges when using reward modeling to train an AI for building a castle in Minecraft.
## Question 2
Stiennon et al. (2020) note that "optimizing our reward model eventually leads to sample quality degradation". Explain why the curves in the corresponding graph are shaped the way they are. How could we prevent performance from decreasing so much?
### My original answer
The reward model is imperfect, from the post "it only sees a small amount of comparison data from a narrow distribution of summaries." When the RL model is optimized against the reward model too intensely, it essentially learns to generate adversarial examples to maximize the reward given by the reward model. These examples take advantage of differences between the true reward and the reward model. TL;DR: essentially just overfitting, with more steps.

The authors address this in one way, by introducing a loss for deviation from a baseline model. Another way that I think could be promising would be to take inspiration from GANs. Have the reward model update more often with respect to the RL model. The RL model would have less time to learn to abuse the reward model before it changed due to new feedback. Perhaps there is also a way to use the RL model to select which pairs of summaries the reviewers rate in such a way as to improve the sample efficiency of the reward model itself too.
### Answer co-written with an LLM
The reward model optimization process described by Stiennon et al. (2020) leads to sample quality degradation due to several factors. First, the reward model has limited exposure to a narrow distribution of comparison data, resulting in an imperfect model. When the RL model is intensely optimized against this imperfect reward model, it can exploit discrepancies between the true reward and the reward model, effectively generating adversarial examples to maximize rewards.

To prevent significant performance decrease, the authors introduce a loss for deviation from a baseline model. Additionally, adopting ideas from Generative Adversarial Networks (GANs) could be promising. For instance, updating the reward model more frequently with respect to the RL model would limit the RL model's ability to exploit the reward model before it receives updated feedback. Another potential approach could involve leveraging the RL model to select pairs of summaries for reviewer ratings, thereby improving the sample efficiency of the reward model.

In summary, the sample quality degradation observed in the optimization process arises from the imperfect reward model and the RL model's ability to exploit it. Mitigating this degradation can be achieved by introducing loss mechanisms, increasing the frequency of reward model updates, and utilizing the RL model to improve the efficiency of the reward model.
## Discussion Prompts
### Prompt 1
What are the key similarities and differences between behavioural cloning, reward modelling, and inverse reinforcement learning?
#### Answer 1
All are attempting to learn human preferences in some way. BC tried to directly learn the behaviour of the human, RM tries to predict the reward a human would give if they were to evaluate the current state, and IRL tries to learn a reward function that represents what the human itself is optimizing for.
### Prompt 2
What types of human preferences can these techniques most easily learn? What types would be hardest to learn?
#### Answer 2
Simple ones will be easier. Extended or complex preferences, especially those relying on subtle features of the context which might not be available to the AI, would be very difficult. It may also be difficult to learn preferences for extremely novel situations (which may themselves be caused by the AI).
### Prompt 3
Should we expect techniques like reward modelling and IRL to be necessary for building AGI (independent of safety concerns)?
#### Answer 3
Independent of safety concerns, I would not expect this is necessary. But something like these will probably be necessary to make useful advanced AI systems.
### Prompt 4
How might using reward modelling lead to misaligned AGIs?
#### Answer 4
Optimizing too much on the reward model. Humans providing incorrect feedback. Humans in disagreement about specific principles. Lack of corrigibility.
