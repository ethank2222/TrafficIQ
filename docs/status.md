---
layout: default
title: Status
---

## Project Summary
**TrafficIQ** is an intelligent traffic light system designed to minimize car idle times at intersections and reduce traffic congestion. The system takes in the number of halted vehicles, average vehicle length, and cumulative waiting time for each of the four approach lanes (North, South, East, West), along with the current traffic light phase and simulation time. Given this information, it decides whether to keep or switch the active green direction (North/South vs. East/West).

## Approach
TrafficIQ utilizes Proximal Policy Optimization (PPO) as its core learning algorithm, built on top of an Actor-Critic network architecture. This traffic light control system, currently, operates in the context of a single 4-way intersection.

The simulation environment is provided by SUMO (Simulation of Urban MObility), an open-source traffic simulator. The environment models a 4-way intersection with bidirectional travel on all lanes. At every timestep, the following features are recorded per lane: total halting vehicle count, mean vehicle length, and cumulative vehicle waiting time. A vehicle is considered halting if its speed falls below 0.1 m/s. Mean vehicle length is the average length of all vehicles currently present in the lane. Cumulative waiting time is the sum of accumulated wait times across all vehicles in a given lane. In addition to these per-lane features, two global values are shared across all lanes: current traffic light phase and current simulation time. Current phase reports the active signal state of the intersection, and simulation time tracks the total elapsed time of the simulation run. Together, these 14 values (4 lanes × 3 features + 2 global) form the state vector passed to the agent at each decision step.

<img src="./assets/images/sumo.png" alt="SUMO Simulation Example" class="home-image">

The 14-dimensional state vector is fed into an Actor-Critic neural network consisting of two shared hidden layers followed by two separate output heads. The shared layers are fully connected, each with 64 neurons, and use a Tanh activation function. Tanh is preferred here over alternatives such as ReLU because it is zero-centered and bounded between (-1, 1), producing smoother, more stable gradients, a meaningful advantage when the critic must estimate values that may be positive or negative (in our case strictly negative). These shared layers act as a common feature extractor, allowing both the actor and critic to learn from the same intermediate representation rather than maintaining entirely separate networks, which encourages complementary learning between the two heads. 

The actor head is a single linear layer mapping from 64 neurons to 2 outputs: one logit per possible action. These logits are passed to a Categorical distribution, which converts them into action probabilities internally. The critic head is similarly a single linear layer, mapping from 64 neurons to a single scalar output representing V(s): the estimated expected discounted return from the current state.

The agent is trained by optimizating a loss function aggregating three components: clipped actor (policy) loss, critic (value) loss, and an entropy bonus. 

The actor loss is defined by PPO's clipped surrogate objective, built from two core components: a probability ratio and an advantage estimate. The probability ratio
$$rt(θ)=πθ(at∣st)/πθold(at∣st)$$
measures how much the current policy has shifted relative to the policy that originally collected the data. The advantage estimate $\hat{A}_t$ measures whether a taken action performed better or worse than expected. In our implementation this is simplified relative to full GAE, we compute a Monte Carlo discounted return and subtract the critic's value estimate as a baseline: 
$$A^t=Rt−V(st)\hat{A}_t = R_t - V(s_t) A^t​=Rt​−V(st​).$$

The final actor loss takes the minimum of the unclipped and clipped ratio, each weighted by the advantage, and computes an expectation over the collected trajectory:

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

Taking the minimum acts as a pessimistic bound, the policy is only allowed to benefit from updates that stay within the trust region defined by the clip, preventing large, destructive policy shifts in a single update. We use $\epsilon = 0.2$ and educate the formulas provided above following the work in *Proximal Policy Optimization Algorithms* (2017) [Schulman et al., 2017](https://arxiv.org/pdf/1707.06347).

The critic loss, in comparison, is much simpler. It is derived by the MSE between predicted state value and observed discounted return. This discounted return calcuated via $R = r * 0.9 * R * (1-d)$. We utalize a **GAMMA** of 0.99 as to heavily incentivies future rewards as apposed to immediate ones. This is 

$$(returns - new~values)^2$$

The entropy bonus is drawn from information theory, measuring the uncertainty of the current policy's action distribution: $H[\pi_\theta] = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)
$. A higher entropy indicates a more exploratory, less deterministic policy. Including it in the loss encourages the agent to maintain exploration and avoid premature convergence.

These three terms combine into a single unified loss:
$$actor~loss + 0.5 * critic~loss - 0.01 * entropy$$
We use a value coefficient of $c_1 = 0.5$ to prevent the critic loss from dominating the actor loss during optimization, and an entropy coefficient of $c_2 = 0.01$ to sustain sufficient exploration without destabilizing the policy.

The agent is trained entirely online through direct interaction with the SUMO simulation, with a singular 4-way intersection simulation. Each training episode runs until all vehicles have cleared the intersection, approximately 520 vehicles per episode, making episode length variable rather than fixed. Over the course of 100 training episodes, the agent collects trajectory data on-policy, meaning experience is gathered under the current policy and discarded after each update. This is a fundamental property of PPO. At the end of each episode, the accumulated trajectory is used to perform 4 gradient update passes (epochs) over the same batch before the buffer is cleared for the next episode. In total, the agent therefore performs 400 gradient updates across training. Decisions are made every 10 simulation steps, meaning each episode produces roughly 520 stored transitions depending on traffic throughput and signal timing.

## Evaluation
To evaluate TrafficIQ's performance, we compare total accumulated vehicle waiting time between our PPO agent and a fixed-timer baseline controller. The baseline operates on a naive 200-step toggle, switching between North/South and East/West green phases at a fixed interval regardless of traffic conditions. This serves as our performance floor. Improvement is expressed as a percentage reduction in total waiting time relative to this baseline, giving a single interpretable quantitative metric for each evaluation run.

Our PPO agent achieves a moderate improvement over the baseline, reducing total accumulated waiting time by ~28.3%. While this falls short of the 30% target set at the outset, it demonstrates that the agent has meaningfully learned to respond to traffic conditions rather than simply replicating fixed-interval behavior. Figure 1 shows the training progress curve across all 100 episodes, plotting per-episode improvement against the baseline alongside a 10-episode moving average to surface the underlying trend. The curve shows a noisy centered near our goal of 30%. This shows that, between episodes, we are takeing drastic steps that can, at times, destroy the model. More Analysis is needed but reducing or $\epsilon = 0.2 \rightarrow \epsilon = 0.1$ may aid in reducing this high variability. 

<img src="./assets/images/training_progress.png" alt="Training Peformance Graph" class="home-image">
*Figure 1*

It is worth noting that the baseline used here is intentionally simple, which makes it a low bar for comparison. A more rigorous future evaluation would benchmark against a more sophisticated fixed-timer controller such as Webster's method, which optimizes phase timings analytically based on traffic flow theory. This would provide a stronger and more meaningful point of comparison, and would better isolate the contribution of learned adaptive behavior over well-engineered deterministic control.

## Remaining Goals and Challenge
Our most immediate goal for the remainder of the quarter is strengthening the evaluation framework. As noted in the Evaluation section, our current baseline is a naive fixed-timer controller, which represents a low bar for comparison. We plan to implement a more sophisticated benchmark, such as Webster's method, which analytically optimizes phase timings based on observed traffic flow. Comparing against this would provide a more meaningful measure of what adaptive, learned behavior actually contributes over well-engineered deterministic control, and would better inform where the model is succeeding and where it still falls short.

Beyond evaluation, our current simulation environment is a deliberate simplification of a real intersection. The most significant limitation is the action space: the agent currently only controls two green phases (North/South and East/West), with no support for left-turn signals or pedestrian crosswalk phases. Introducing these would considerably expand the action space and more closely reflect the complexity of real-world traffic light systems. Alongside this, we plan to develop additional simulation environments of varying layouts and traffic densities to train and test across, with the goal of evaluating whether the agent can generalize beyond the single intersection it was trained on.

Our stretch goal is multi-intersection communication, training agents across a network of intersections that can coordinate signal timing with one another, which represents a much harder and more realistic problem. We anticipate the primary challenges to be environment complexity and training instability: as the action and state spaces grow with additional signal phases and intersections, PPO may require significant hyperparameter retuning and longer training to converge. Our fallback is to ensure the single-intersection system is robustly evaluated and clearly documented before pursuing further scope.`

## Resources Used
Mention all the resources that you found useful in implementing your method, experiments, and analysis. This should include everything like code documentation, AI/ML libraries, source code that you used, StackOverflow, etc. You do not have to include every tiny (or commonplace) thing you used, but it is important to report the sources that are crucial to your project. One aspect that does need to be comprehensive is a description of any use you made of AI tools

We utilized <a href="https://sumo.dlr.de/docs/index.html">SUMO documentation</a> to aid initial setup of our intersection enviornment as well as for our light system interface. In addition we utalize documenation for PyTorch and Numpy. Our team as well leveraged online video tutorials present on Youtube.

Our team made substantial use of large language models, primarily Claude and Gemini, across three areas of the project. First, during model development, LLMs were used as a sanity check, helping verify our understanding of PPO concepts and clarifying unfamiliar ideas as they arose. Second, LLMs were used to generate template code that was then reviewed and tailored to the specific needs of our implementation. Third, a significant portion of this written documentation was refined with LLM assistance, including grammar and clarity edits and generation of LaTeX equations where appropriate.

In the future we hope to leverage the work of *Navtegh* and his github repository <a href="https://github.com/Navtegh/Traffic-Light-Management-system-using-RL-and-SUMO">Traffic-Light-Management-system-using-RL-and-SUMO</a>. This will provide invaluable insight in training our initial model and how to further push its complexity.
