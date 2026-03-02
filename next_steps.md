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

### Ethan
- Update hyperparameters of PPO Model to help model converge
    - Ex: $\epsilon = 0.2 \rightarrrow \epsilon = 0.1$ 
        - might help with huge steps ruining the model at times

### Ryan
- Update action space of model for **turn signals** and **pedestrian crossing**
    - Mostly looking into SUMO and traci documenation probs

### Erick
- Create new evaluation model
    - Websters will most likely be best