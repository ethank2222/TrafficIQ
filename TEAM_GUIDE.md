# Team Guide - Essential Info Only

## 🎯 What We're Building

An AI traffic controller that **minimizes total car wait times** at a 4-way intersection using PPO (Proximal Policy Optimization).

**Goal**: 30% reduction in wait time vs baseline fixed-timing

---

## 📁 Essential Files (What to Keep)

```
TrafficIQ/
├── run_sim.py                      # Basic simulation (from Erick)
├── ai_traffic_controller_template.py  # Template for our AI
├── simulations/Easy_4_Way/         # Traffic scenario files
│   ├── map.net.xml                 # Road network
│   ├── map.sumocfg                 # Configuration
│   └── routes.rou.xml              # 400 cars, 4 directions
├── requirements.txt                # Dependencies: traci, sumolib
├── README.md                       # Project overview
└── SUMO_SETUP_GUIDE.md            # Detailed setup instructions
```

---

## 🚀 Quick Setup (Everyone Do This)

### 1. Install SUMO
- Download: https://www.eclipse.org/sumo/
- ✅ Check "Add to PATH" during install

### 2. Set Environment Variable
- Windows: Add user variable `SUMO_HOME` = `C:\Program Files (x86)\Eclipse\Sumo`
- Restart terminal/IDE after setting

### 3. Install Python Packages
```bash
pip install -r requirements.txt
```

### 4. Test It Works
```bash
python run_sim.py
```
Press Spacebar to start, press `-` to slow down

---

## 📊 KEY METRIC: Total Wait Time

**This is what we're optimizing:**

```python
# At each step, sum all car wait times
lanes = ["NI_0", "SI_0", "EI_0", "WI_0"]
total_wait_time = sum(traci.lane.getWaitingTime(lane) for lane in lanes)

# Reward = negative wait time (higher is better)
reward = -total_wait_time
```

**Why this metric?**
- ✅ Measures actual driver frustration
- ✅ Changes with every AI decision
- ✅ What research papers use
- ❌ NOT completion time (that's fixed)

---

## 🤖 AI Integration Structure

### State (14 values):
```python
[
    waiting_cars_N, waiting_cars_S, waiting_cars_E, waiting_cars_W,  # 4
    queue_length_N, queue_length_S, queue_length_E, queue_length_W,  # 4
    wait_time_N, wait_time_S, wait_time_E, wait_time_W,              # 4
    current_phase,                                                     # 1
    simulation_time                                                    # 1
]
```

### Actions (2 choices):
- **0**: North-South Green
- **2**: East-West Green

### Reward:
- **reward = -total_wait_time** (minimize waiting!)

---

## 🧪 How to Test

### Get Baseline Number:
```bash
python ai_traffic_controller_template.py
```
This runs the baseline (fixed 200s timing) and shows total wait time.

**Expected**: ~60,000-65,000 seconds total wait time

### Add Your AI:
Edit `ai_traffic_controller_template.py`:
1. Import your PPO agent
2. Replace the TODOs in `run_your_ai()` function
3. Run again and compare!

---

## 📈 Evaluation Plan

### Baselines to Beat:
1. **Fixed timing** (200s cycles) ← Start here
2. **Time-of-day** (adjust by traffic)
3. **Sensor-reactive** (extend green if queue exists)

### Target Improvements:
- vs Fixed: **30% reduction**
- vs Time-of-day: **10-20% reduction**
- vs Reactive: **10-20% reduction**

### Metrics to Report:
- **Primary**: Average wait time per vehicle
- Secondary: Total wait time, throughput, queue lengths

---

## 💻 Code Pattern for AI

```python
# Training loop structure
env = TrafficEnvironment()
env.start()

while not env.is_done():
    # 1. Observe
    state = env.get_state()  # 14 values
    
    # 2. Decide
    action = your_ppo_agent.select_action(state)  # 0 or 2
    
    # 3. Act
    env.take_action(action)
    env.step()
    
    # 4. Evaluate
    reward = env.get_reward()  # -total_wait_time
    next_state = env.get_state()
    
    # 5. Learn
    your_ppo_agent.store(state, action, reward, next_state)
    if ready:
        your_ppo_agent.update()

env.close()
```

---

## 🎮 SUMO Controls (While Running)

- **Spacebar**: Start/Pause
- **`-`**: Slow down (good for watching)
- **`+`**: Speed up
- **`S`**: Step one frame at a time
- Click on cars to see their details

---

## 📞 Questions?

1. **Setup issues?** Check `SUMO_SETUP_GUIDE.md`
2. **How does SUMO work?** Run `run_sim.py` and watch
3. **What data is available?** Look at `get_state()` in template
4. **Metric confusion?** We use **sum of wait times**, not completion time

---

## ✅ Next Steps

**Week 1 (This Week):**
- [ ] Everyone: Set up SUMO and run `run_sim.py`
- [ ] Understand baseline performance
- [ ] Decide on work split

**Week 2:**
- [ ] Implement simple rule-based controller (better than fixed timing)
- [ ] Set up PPO framework

**Week 3-4:**
- [ ] Train PPO agent
- [ ] Implement baselines
- [ ] Run evaluations

**Week 5:**
- [ ] Final testing
- [ ] Report writing
- [ ] (Stretch) Multi-intersection

---

## 🔑 Key Takeaway

**Minimize this number**: `sum(traci.lane.getWaitingTime(lane) for all lanes)`

That's it! Everything else is about how to measure this and how to teach the AI to optimize it.

