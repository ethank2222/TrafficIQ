---
layout: default
title:  Home
---

Source code: https://github.com/ethank2222/TrafficIQ

Reports:

- [Proposal](proposal.html)
- [Status](status.html)
- [Final](final.html)

**Project Summary**

- Overview: The `rl_traffic_agent.py` script implements a Proximal Policy Optimization (PPO) agent that controls a single traffic light in a SUMO simulation (Easy_4_Way map). It defines an Actor-Critic network, environment interaction via TraCI, and a PPO training loop that runs for a configurable number of episodes and compares performance against a simple fixed-phase baseline.
- Outputs: Trained model saved as `ppo_traffic_model.pt` and a training progress plot saved as `training_progress.png`.
- Goal: Reduce average vehicle waiting time (the script targets a 30% improvement vs baseline).

**Example Training Run**

![Example Training Run](./training_progress.png)

**Example Training Run**

![Our Hyperparameters](./setting_hyperparameters.png)

**Runtime Logging/Messages**

![Runtime Logging/Messages](./run_messages.png)
