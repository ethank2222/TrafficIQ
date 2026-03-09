Better light logic (right now uses sumo defaults, should have only only straight/right turns) - add 2 dimensions -> from 2 to 4
Pedestrians
Multiple Intersections
Add more than 1 lane per direction
Currently is a uniform arrival rate
Other models for reference?

Baseline:

What it does:

- Every 200 simulation steps, toggle the traffic light
- If currently N-S green (phase 0) → switch to E-W green (phase 2)
- If currently E-W green (phase 2) → switch to N-S green (phase 0)
- Completely ignores traffic conditions (no sensors, no intelligence)


## TASKS

### IMPROVE MODEL
*Rough Notes from our meeting with Prof. Fox*
- Intermediate signals for reward
    - Helps for credit assignment
- FOCUS ON SHAPING REWARD
    - can tally all rewards, then give it some difference 
    - Largely to help shape it
- Should mimik behavior what we want
    - i.e not leaving a car in red indefintly so all other cars can pass
- Choose final hyper parameters
    - Already have code for this so I can just help pick when we reshape reward (Erick)

--- 

### CREATE SLIDES
1. Motivation
2. Problem statement
3. Approach / method
4. Results
5. Discussion / insight
- Need Video demonstrating our model at work
    - I can finnish my code to make this easy (Erick)
- Can copy over a LOT from our status.md

