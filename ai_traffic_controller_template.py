"""
SIMPLIFIED AI Template - Track Total Wait Times
Essential code for your CS175 AI Traffic Controller Project

KEY METRIC: Sum of all car wait times (minimize this!)
"""

import os
import sys
import traci

# TODO: Import your PPO implementation when ready
# from your_ppo_agent import PPOAgent

CONFIG_PATH = os.path.join("simulations", "Easy_4_Way", "map.sumocfg")
TL_ID = "TCenter"

# Setup SUMO_HOME
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please set SUMO_HOME environment variable!")


class TrafficEnvironment:
    """
    Simple SUMO wrapper for your AI
    Main feature: Tracks total wait times (your key metric!)
    """
    
    def __init__(self, config_path=CONFIG_PATH, use_gui=True):
        self.config_path = config_path
        self.use_gui = use_gui
        self.step_count = 0
        
        # Lane IDs for the 4-way intersection
        self.lanes = ['NI_0', 'SI_0', 'EI_0', 'WI_0']  # N, S, E, W
        
    def start(self):
        """Start SUMO simulation"""
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        traci.start([sumo_binary, "-c", self.config_path])
        self.step_count = 0
        
    def get_state(self):
        """
        Get current traffic state for your AI
        
        Returns:
            list: [waiting_counts (4), queue_lengths (4), wait_times (4), phase (1), time (1)]
                  Total: 14 values
        """
        state = []
        
        # Waiting vehicles per lane (4 values)
        for lane in self.lanes:
            state.append(traci.lane.getLastStepHaltingNumber(lane))
            
        # Queue lengths in meters (4 values)
        for lane in self.lanes:
            state.append(traci.lane.getLastStepLength(lane))
            
        # Wait times per lane (4 values)
        for lane in self.lanes:
            state.append(traci.lane.getWaitingTime(lane))
            
        # Current phase (1 value)
        state.append(traci.trafficlight.getPhase(TL_ID))
        
        # Simulation time (1 value)
        state.append(traci.simulation.getTime())
        
        return state
    
    def take_action(self, action):
        """
        Switch traffic light
        
        Args:
            action: 0 = North-South Green, 2 = East-West Green
        """
        traci.trafficlight.setPhase(TL_ID, action)
        
    def get_total_wait_time(self):
        """
        **KEY METRIC**: Total wait time across all lanes
        This is what your AI should minimize!
        
        Returns:
            float: Sum of all car wait times in seconds
        """
        return sum(traci.lane.getWaitingTime(lane) for lane in self.lanes)
    
    def get_reward(self):
        """
        Reward for your AI = negative wait time
        Lower wait time = Higher reward = Better!
        """
        return -self.get_total_wait_time()
    
    def step(self):
        """Advance simulation by one time step"""
        traci.simulationStep()
        self.step_count += 1
        
    def is_done(self):
        """Check if simulation is complete"""
        return traci.simulation.getMinExpectedNumber() <= 0
    
    def close(self):
        """Close SUMO simulation"""
        traci.close()
        
    def get_metrics(self):
        """Performance metrics for evaluation"""
        return {
            'total_wait_time': self.get_total_wait_time(),
            'num_vehicles': traci.vehicle.getIDCount(),
            'simulation_time': traci.simulation.getTime()
        }


def run_baseline():
    """
    Baseline test: Fixed timing (switches every 200 steps)
    Run this to get your baseline wait time number
    """
    env = TrafficEnvironment(use_gui=True)
    env.start()
    
    cumulative_wait_time = 0
    
    print("\n" + "="*70)
    print("BASELINE: Fixed timing (200 second intervals)")
    print("="*70)
    
    while not env.is_done():
        # Simple fixed timing
        if env.step_count % 200 == 0:
            current_phase = traci.trafficlight.getPhase(TL_ID)
            new_phase = 2 if current_phase == 0 else 0
            env.take_action(new_phase)
            print(f"Step {env.step_count}: Switched to {'EW' if new_phase == 2 else 'NS'} green")
        
        env.step()
        # Accumulate wait time at EACH step (not just at the end!)
        cumulative_wait_time += env.get_total_wait_time()
    
    print(f"\n{'='*70}")
    print(f"BASELINE RESULTS:")
    print(f"  Total Wait Time: {cumulative_wait_time:.2f} seconds")
    print(f"  Avg per car:     {cumulative_wait_time/400:.2f} seconds")
    print(f"{'='*70}\n")
    
    env.close()
    return cumulative_wait_time


def run_your_ai():
    """
    TODO: Replace this with your actual PPO implementation
    
    This is a placeholder showing the structure
    """
    env = TrafficEnvironment(use_gui=True)
    env.start()
    
    # TODO: Initialize your PPO agent here
    # agent = YourPPOAgent(state_dim=14, action_dim=2)
    
    cumulative_wait_time = 0
    
    print("\n" + "="*70)
    print("YOUR AI: (currently just a simple rule)")
    print("="*70)
    
    while not env.is_done():
        # Get state
        state = env.get_state()
        
        # TODO: Replace with your AI's decision
        # action = agent.select_action(state)
        
        # Placeholder: Simple reactive rule
        waiting_ns = state[0] + state[1]
        waiting_ew = state[2] + state[3]
        current_phase = state[13]
        
        if env.step_count % 50 == 0:  # Check every 50 steps
            if current_phase == 0 and waiting_ew > waiting_ns + 3:
                env.take_action(2)
            elif current_phase == 2 and waiting_ns > waiting_ew + 3:
                env.take_action(0)
        
        # Get reward
        reward = env.get_reward()
        
        # TODO: Train your agent
        # agent.store_experience(state, action, reward, next_state)
        
        env.step()
        # Accumulate wait time at EACH step (not just at the end!)
        cumulative_wait_time += env.get_total_wait_time()
    
    print(f"\n{'='*70}")
    print(f"YOUR AI RESULTS:")
    print(f"  Total Wait Time: {cumulative_wait_time:.2f} seconds")
    print(f"  Avg per car:     {cumulative_wait_time/400:.2f} seconds")
    print(f"{'='*70}\n")
    
    env.close()
    return cumulative_wait_time


if __name__ == "__main__":
    print("\n*** AI Traffic Controller Test ***\n")
    
    # Run baseline
    baseline = run_baseline()
    
    # Run your AI
    print("\n(Press Enter to run your AI version)")
    input()
    ai_result = run_your_ai()
    
    # Compare
    if baseline > 0:
        improvement = ((baseline - ai_result) / baseline) * 100
        print("\n" + "="*70)
        print("COMPARISON:")
        print(f"  Baseline:    {baseline:.2f}s")
        print(f"  Your AI:     {ai_result:.2f}s")
        print(f"  Improvement: {improvement:.1f}%")
        print(f"  Goal:        30% improvement")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("ERROR: No wait time detected!")
        print("Make sure simulation ran and cars spawned properly.")
        print("="*70)




