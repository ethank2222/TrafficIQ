# AI Traffic Signal Controller

CS175 Project - Intelligent Traffic Light Optimization using Reinforcement Learning

## 🚦 Project Overview

This project develops an AI-powered traffic signal controller that minimizes vehicle wait times at intersections using Proximal Policy Optimization (PPO). Unlike traditional fixed-timing signals, our system adapts in real-time based on actual traffic conditions.

## 🎯 Project Goals

- **Minimum**: 4-way intersection with vehicle-only traffic optimization
- **Realistic**: 4-way intersection with vehicles and pedestrians
- **Moonshot**: N-way intersections (complex geometries like 6-way, one-way streets)

**Target**: 30% reduction in average wait time compared to fixed-timing signals

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- SUMO (Simulation of Urban MObility)

### Setup
```bash
# 1. Install SUMO from https://www.eclipse.org/sumo/
# 2. Set SUMO_HOME environment variable
# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Run baseline simulation
python run_sim.py
```

📖 **Detailed setup instructions**: See `QUICK_START.md` or `SUMO_SETUP_GUIDE.md`

## 📁 Project Structure

```
project-ree/
├── simulations/
│   └── Easy_4_Way/          # 4-way intersection scenario
│       ├── map.net.xml      # Road network definition
│       ├── map.sumocfg      # SUMO configuration
│       └── routes.rou.xml   # Vehicle traffic flows
├── run_sim.py               # Baseline fixed-timing controller
├── ai_traffic_controller_template.py  # Template for AI integration
├── requirements.txt         # Python dependencies
├── QUICK_START.md          # 5-minute setup guide
├── SUMO_SETUP_GUIDE.md     # Comprehensive documentation
└── docs/                   # Project website
```

## 🤖 How It Works

1. **SUMO** simulates realistic traffic at a 4-way intersection
2. **TraCI** (Traffic Control Interface) connects Python to SUMO
3. **Your AI** observes traffic state (queue lengths, wait times, etc.)
4. **PPO Agent** learns optimal light-switching policy
5. **Reward**: Negative total wait time (minimize waiting!)

### State Space (14 dimensions)
- Waiting vehicles per lane (4)
- Queue lengths (4)
- Average wait times (4)
- Current light phase (1)
- Time since last change (1)

### Action Space (4 actions)
- North-South Green
- North-South Yellow
- East-West Green
- East-West Yellow

## 📊 Evaluation Metrics

**Primary**: Average wait time per vehicle

**Baselines**:
1. Fixed timing (200s cycles)
2. Time-of-day adjusted
3. Basic sensor-reactive

**Expected Improvements**:
- vs Fixed: 30% reduction
- vs Time-of-Day: 10-20% reduction
- vs Sensor-Reactive: 10-20% reduction

## 🛠️ Technologies

- **SUMO**: Traffic simulation engine
- **Python + TraCI**: Control interface
- **PPO (Proximal Policy Optimization)**: Reinforcement learning algorithm
- **NumPy**: Numerical computations

## 📚 Documentation

- **Quick Start**: `QUICK_START.md` - Get running in 5 minutes
- **Full Guide**: `SUMO_SETUP_GUIDE.md` - Detailed explanations
- **AI Template**: `ai_traffic_controller_template.py` - Integration example
- **Project Website**: See `docs/` folder

## 🔗 Useful Resources

- [SUMO Documentation](https://sumo.dlr.de/docs/)
- [TraCI API Reference](https://sumo.dlr.de/docs/TraCI.html)
- [PPO Paper](https://arxiv.org/abs/1707.06347)

## 👥 Team

See `docs/team.md` for team information.

## 📝 License

CS175 Course Project - Academic Use Only