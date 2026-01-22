---
layout: default
title:  Proposal
---

## Project Summary
Our AI signal system is an intelligent traffic signal that minimizes idle time at intersections. It uses contextual information including the total number of cars, lane occupancy, time of day, and pedestrian presence. Unlike traditional traffic signals, it makes informed decisions based on historical data and real-time context to minimize wait times without compromising safety.

## Project Goals

**Minimum Goal:**
Our minimum goal is to create a simulation that models cars only arriving at a 4-way traffic light controlled intersection over the course of a week. Based on this simulation, our AI algorithm will continually learn and optimize the light schedule for each direction, including left turn and straight green lights.

**Realistic Goal:**
Our realistic goal is to create a simulation that models cars and pedestrians arriving at a 4-way traffic light controlled intersection over the course of a week. Based on this simulation, our AI algorithm will continually learn and optimize the light and crossing schedules for each direction, including left turn and straight green lights.

**Moonshot Goal:**
Our moonshot goal is to create a simulation that models cars and pedestrians arriving at an n-way traffic light controlled intersection over the course of a week. This intersection could have 1-way streets, or even be a 6-way intersection. Based on any situation, our AI algorithm would continually learn and optimize the light schedule and crossing schedule for each direction.

## AI/ML Algorithms
Tentatively, we will use Proximal Policy Optimization (PPO), a model-free, on-policy reinforcement learning algorithm with a neural network function approximator.

## Evaluation Plan
Quantitative Eval: For evaluating the success of our AI traffic signal system, we will conduct experiments comparing our AI-based approach against several baseline methods. Our primary metric will be the average total wait time per vehichle across all lanes at the intersection, measured over different lengths of time. Some naive approaches we will use as our baselines will consist of, a fixed time signal that cycles through each direction on equal intervals, a time of day based system that will change its cycle depending on rush hour traffic or very small traffic, and a basic sensor-reactive system that will keep a green light on if it detects a long queue of cars. We will run multiple simulations against all 3 with different envioenrment variables, such as number of intersections, allowing pedestrians, and increasing the queue of cars. For our AI, we hope to reduce wait time against all 3 of these baselines. Against the fixed cycle, we hope to see the highest decrease in wait time with around 30%. However against the other two baselines since they are reactive to different variables, we hope our AI will perform around 10-20% better.

Qualitative Eval: For qualitative verification, we will implement several debugging and visualization tools to ensure our AI system behaves safely and logically. For our toy cases, will start with basic and controlled scenarios, such as a single car going in one direction, even traffic from all directions, and an intersection with a higher concentration of cars in one lane. We expect our AI to react to these situaitons appropriately and change the traffic light colors to decrease the wait time for all the cars. Visualizing the states are important for helping our AI learn, some states being the veichle queues, current signal states, and AI's previous decision actions. For success in this area, our AI needs to make sure that no saftely violations are broken, there is no bias on a specific intersection lane, responsiveness to changes are picked up quickly, and that there are smooth transitions that avoid unnecessary signal switching. For edge cases, we will manually test those scenarios, such as heavy pedestrian presence, or an overflow of traffic in multiple lanes in an intersection. For tracking interal results, Decision trace logs showing why the AI took each action and feature importance graphs can be visualized to show which factors have the most influence on our AI's decision making. For external results, we can visaully confirm with a simulation to display how our traffic light AI is responding.



## AI Tool Usage
Our plan is to use AI tools to help us create the testing environment/simulation engine. We will be creating the ai algorithm for the lights and crossing signs, but we want a fairly constantly random assortment of pedestrians and cars coming to our intersection. AI tools will help us model this to give us a good testing environment

