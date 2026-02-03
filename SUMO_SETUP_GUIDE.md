# SUMO Setup Guide for AI Traffic Controller Project

## 📖 Overview

This guide will help you set up and use SUMO (Simulation of Urban MObility) for your CS175 AI Traffic Controller project. SUMO provides a realistic traffic simulation environment where your AI can learn to optimize traffic light timing.

---

## 🔧 Installation & Setup

### Step 1: Install SUMO

**Windows Installation:**
1. Download from: https://www.eclipse.org/sumo/
2. Run the installer (version 1.25.0 recommended)
3. ✅ **Important**: Check the box to add SUMO to system PATH during installation
4. Default installation path: `C:\Program Files (x86)\Eclipse\Sumo`

**Verify Installation:**
```powershell
sumo --version
```
Expected output: `Eclipse SUMO sumo Version 1.25.0`

---

### Step 2: Set SUMO_HOME Environment Variable

**Windows:**
1. Press `Win + R`, type `sysdm.cpl`, press Enter
2. Go to "Advanced" tab → "Environment Variables"
3. Under "System variables", click "New"
4. Set:
   - **Variable name:** `SUMO_HOME`
   - **Variable value:** `C:\Program Files (x86)\Eclipse\Sumo` (or your installation path)
5. Click "OK" to save
6. **Restart your terminal/IDE**

**Verify:**
```powershell
echo $env:SUMO_HOME
```

---

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `traci`: Python interface to control SUMO
- `sumolib`: Utilities for working with SUMO files

---

## 🎮 Running Your First Simulation

### Test the Basic Simulation

```bash
python run_sim.py
```

**What you'll see:**
- GUI window with a 4-way intersection
- Cars spawning from all four directions
- Traffic lights changing every 200 seconds
- Console output showing phase switches

**Controls in SUMO GUI:**
- **Play/Pause**: Space bar
- **Speed up**: `+` key
- **Slow down**: `-` key
- **Zoom**: Mouse wheel
- **Pan**: Right-click drag

---

## 📁 Understanding Your Simulation Files

### `simulations/Easy_4_Way/`

#### 1. `map.net.xml` - Road Network
Defines the physical layout:
- **Intersection ID**: `TCenter` (at coordinates 0, 0)
- **4 Roads**: North, South, East, West (each 400m long)
- **Traffic Light Phases**:
  - Phase 0: North-South Green (42s)
  - Phase 1: North-South Yellow (3s)
  - Phase 2: East-West Green (42s)
  - Phase 3: East-West Yellow (3s)

#### 2. `routes.rou.xml` - Traffic Patterns
Defines vehicle flows:
- **100 cars** from North → South
- **100 cars** from South → North
- **100 cars** from East → West
- **100 cars** from West → East
- Total: **400 cars** over 1000 seconds

**Vehicle Properties:**
- Acceleration: 2.6 m/s²
- Deceleration: 4.5 m/s²
- Max speed: 13.89 m/s (≈50 km/h)
- Length: 5.0 m

#### 3. `map.sumocfg` - Configuration
Links everything together:
- Network file: `map.net.xml`
- Routes file: `routes.rou.xml`
- Simulation time: 0-1000 seconds

---

## 🤖 Integrating Your AI

### Understanding the TraCI Interface

**TraCI** (Traffic Control Interface) lets your Python code control SUMO in real-time.

### Key Functions for Your AI

#### 1. **Get State Information** (for observations)

```python
import traci

# Number of waiting/stopped vehicles
waiting_cars = traci.lane.getLastStepHaltingNumber("NI_0")  # North lane

# Queue length in meters
queue_length = traci.lane.getLastStepLength("NI_0")

# Total wait time accumulated on a lane
wait_time = traci.lane.getWaitingTime("NI_0")

# Average speed on lane
avg_speed = traci.lane.getLastStepMeanSpeed("NI_0")

# Current traffic light phase
current_phase = traci.trafficlight.getPhase("TCenter")

# Total vehicles in simulation
num_vehicles = traci.vehicle.getIDCount()

# Simulation time (in seconds)
sim_time = traci.simulation.getTime()
```

#### 2. **Take Actions** (control traffic lights)

```python
# Set to a specific phase (0, 1, 2, or 3)
traci.trafficlight.setPhase("TCenter", phase_number)

# Or set state directly (advanced)
traci.trafficlight.setRedYellowGreenState("TCenter", "GGgrrrGGgrrr")
```

#### 3. **Calculate Rewards**

```python
# Total wait time (minimize this!)
lanes = ["NI_0", "SI_0", "EI_0", "WI_0"]
total_wait = sum(traci.lane.getWaitingTime(lane) for lane in lanes)

# Reward = negative wait time (higher is better)
reward = -total_wait
```

---

## 🏗️ Building Your AI Controller

### Step-by-Step Approach

#### Phase 1: Understanding (Week 1)
1. ✅ Run `run_sim.py` to see baseline behavior
2. ✅ Modify the timing (e.g., change 200 to 100) and observe
3. ✅ Add print statements to see state information

#### Phase 2: Simple Rule-Based Controller (Week 2)
```python
# Example: Switch if other direction has more waiting cars
if waiting_north_south > waiting_east_west + 5:
    switch_to_ns_green()
else:
    switch_to_ew_green()
```

Test this against fixed timing baseline!

#### Phase 3: PPO Implementation (Weeks 3-4)

**State Space (14 dimensions):**
- Waiting vehicles per lane (4)
- Queue lengths per lane (4)
- Wait times per lane (4)
- Current phase (1)
- Time since last change (1)

**Action Space (4 actions):**
- Action 0: North-South Green
- Action 1: North-South Yellow
- Action 2: East-West Green
- Action 3: East-West Yellow

**Reward:**
```python
reward = -total_wait_time  # Minimize wait time
```

**Training Loop:**
```python
for episode in range(1000):
    env.start()
    while not env.is_done():
        state = env.get_state()
        action = agent.select_action(state)
        env.take_action(action)
        env.step()
        reward = env.get_reward()
        next_state = env.get_state()
        
        agent.store_transition(state, action, reward, next_state)
        
        if ready_to_update:
            agent.update()
    env.close()
```

---

## 📊 Evaluation Plan (from your proposal)

### Baselines to Compare Against

1. **Fixed Timing** (current `run_sim.py`)
   - Switches every 200 seconds
   - No awareness of traffic

2. **Time-of-Day Based**
   - Longer green for heavy direction during rush hour
   - Still not reactive to real-time traffic

3. **Basic Sensor-Reactive**
   - Extends green if queue is long
   - No learning or optimization

### Metrics to Track

**Primary Metric:**
- **Average wait time per vehicle** (minimize this!)

**Secondary Metrics:**
- Average speed (higher is better)
- Queue lengths (shorter is better)
- Throughput (vehicles/hour)

### Expected Improvements
- vs Fixed Timing: **~30% reduction** in wait time
- vs Time-of-Day: **~10-20% reduction**
- vs Basic Reactive: **~10-20% reduction**

---

## 🧪 Testing Strategy

### 1. Toy Cases (Sanity Checks)
```python
# Test 1: Single car from North
# Expected: Should quickly give green to North

# Test 2: Even traffic from all directions  
# Expected: Should alternate fairly

# Test 3: Heavy traffic from North only
# Expected: Should favor North direction
```

### 2. Modify Traffic Patterns

Edit `routes.rou.xml`:
```xml
<!-- Heavy North-South traffic -->
<flow id="flow_ns" route="n_s" begin="0" end="1000" number="300"/>
<flow id="flow_sn" route="s_n" begin="0" end="1000" number="300"/>
<flow id="flow_ew" route="e_w" begin="0" end="1000" number="50"/>
<flow id="flow_we" route="w_e" begin="0" end="1000" number="50"/>
```

### 3. Edge Cases
- Rush hour (increase `number` in flows)
- Unbalanced traffic (favor one direction)
- Low traffic (decrease `number`)

---

## 🛠️ Useful Tips & Tricks

### Run Without GUI (Faster Training)
```python
traci.start(["sumo", "-c", CONFIG_PATH])  # Remove "-gui"
```

### Collect Data for Analysis
```python
import csv

with open('results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['step', 'wait_time', 'queue_length', 'phase'])
    
    while not env.is_done():
        # ... your simulation code ...
        writer.writerow([step, wait_time, queue_length, phase])
```

### Debug Visualization
```python
# Print state at each decision point
print(f"State: {state}")
print(f"Action taken: {action}")
print(f"Reward: {reward}")
```

### Speed Up Training
```python
# Run multiple simulations in parallel (advanced)
# Use "sumo" instead of "sumo-gui"
# Reduce simulation end time for faster episodes
```

---

## 🎯 Next Steps for Your Team

1. **Everyone**: Set up SUMO_HOME and run `run_sim.py`
2. **Everyone**: Read through `ai_traffic_controller_template.py`
3. **Divide work**:
   - Person A: Implement PPO agent
   - Person B: Create baseline algorithms (time-of-day, sensor-reactive)
   - Person C: Build evaluation/visualization tools
   - Person D: Design different traffic scenarios

4. **Week by week**:
   - Week 1: Setup + understand SUMO
   - Week 2: Simple rule-based controller
   - Week 3: PPO implementation
   - Week 4: Training & evaluation
   - Week 5: Multi-intersection (moonshot goal)

---

## 📚 Resources

- **SUMO Documentation**: https://sumo.dlr.de/docs/
- **TraCI Tutorial**: https://sumo.dlr.de/docs/TraCI.html
- **PPO Paper**: https://arxiv.org/abs/1707.06347
- **Similar Projects**: Search "SUMO reinforcement learning traffic"

---

## ❓ Common Issues

### Issue: "Please declare environment variable 'SUMO_HOME'"
**Solution**: Set SUMO_HOME as described in Step 2, restart terminal

### Issue: SUMO GUI doesn't open
**Solution**: Check if SUMO is installed, try `sumo --version`

### Issue: No cars appearing
**Solution**: Check `routes.rou.xml` exists and has valid flow definitions

### Issue: Simulation ends immediately
**Solution**: Check simulation time in `map.sumocfg` (should be 1000)

---

## 🎉 You're Ready!

You now have:
- ✅ SUMO installed and configured
- ✅ Understanding of how traffic simulation works
- ✅ Template code for AI integration
- ✅ Evaluation plan

**Start with**: Run `python run_sim.py` and watch the baseline in action!

Good luck with your CS175 project! 🚦🤖




