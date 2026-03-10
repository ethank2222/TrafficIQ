"""
PPO Traffic Light Controller with Enhanced Training

Improvements over baseline:
1. Enhanced reward function:
   - Penalizes wait time (primary)
   - Penalizes excessive switching (yellow light waste)
   - Rewards throughput (completed journeys)
   - Penalizes queue imbalance (fairness)

2. Richer state representation (18 dims):
   - Queue lengths per lane (4)
   - Occupancy per lane (4)
   - Wait times per lane (4)
   - Rate of change in queues (4) - NEW!
   - Current phase (1)
   - Time since last switch (1) - NEW!

3. Larger network capacity:
   - 128->128->64 neurons (vs 64->64)
   - Better capacity for complex patterns

4. Training stability improvements:
   - Reward normalization
   - Lower learning rate (2e-4 vs 3e-4)
   - Learning rate scheduler
   - Early stopping with best model checkpointing
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("Please set SUMO_HOME environment variable!")

import traci

CONFIG_PATH = os.path.join("simulations", "Easy_4_Way", "map.sumocfg")
TL_ID = "TCenter"
LANES = ['NI_0', 'SI_0', 'EI_0', 'WI_0']

STATE_DIM = 18  # Increased from 14
ACTION_DIM = 2
DECISION_INTERVAL = 10
YELLOW_DURATION = 3
NUM_EPISODES = 200  # Increased from 100 for better convergence
GAMMA = 0.99
LR = 2e-4  # Reduced from 3e-4 for more stable training
CLIP_EPS = 0.15
PPO_EPOCHS = 4
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),  # Increased from 64
            nn.Tanh(),
            nn.Linear(128, 128),  # Increased from 64
            nn.Tanh(),
            nn.Linear(128, 64),  # Added extra layer
            nn.Tanh(),
        )
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        features = self.shared(x)
        return self.actor(features), self.critic(features)

    def get_action(self, state):
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze(-1)

    def evaluate(self, states, actions):
        logits, values = self.forward(states)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), values.squeeze(-1), dist.entropy()


class PPOAgent:
    def __init__(self):
        self.network = ActorCritic(STATE_DIM, ACTION_DIM)
        self.optimizer = optim.Adam(self.network.parameters(), lr=LR)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.9)
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        # Running statistics for reward normalization
        self.reward_mean = 0
        self.reward_std = 1
        self.reward_history = []

    def select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, value = self.network.get_action(state_t)
        return action, log_prob, value

    def store(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def update(self):
        if len(self.states) == 0:
            return

        # Update reward statistics for normalization
        self.reward_history.extend(self.rewards)
        if len(self.reward_history) > 1000:  # Keep last 1000 rewards
            self.reward_history = self.reward_history[-1000:]
        if len(self.reward_history) > 10:
            self.reward_mean = np.mean(self.reward_history)
            self.reward_std = np.std(self.reward_history) + 1e-8

        # Normalize rewards for more stable training
        normalized_rewards = [(r - self.reward_mean) / self.reward_std for r in self.rewards]

        returns = []
        R = 0
        for r, d in zip(reversed(normalized_rewards), reversed(self.dones)):
            R = r + GAMMA * R * (1 - d)
            returns.insert(0, R)

        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.stack(self.log_probs).detach()
        returns = torch.FloatTensor(returns)
        old_values = torch.stack(self.values).detach()

        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Diagnostic: Track learning metrics
        self.last_value_loss = None
        self.last_policy_loss = None
        self.last_entropy = None

        for _ in range(PPO_EPOCHS):
            new_log_probs, new_values, entropy = self.network.evaluate(states, actions)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - new_values).pow(2).mean()
            entropy_bonus = entropy.mean()

            loss = actor_loss + VALUE_COEF * critic_loss - ENTROPY_COEF * entropy_bonus

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
            
            # Store last epoch's losses for diagnostics
            self.last_policy_loss = actor_loss.item()
            self.last_value_loss = critic_loss.item()
            self.last_entropy = entropy_bonus.item()

        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
    
    def get_action_distribution(self, state):
        """Get probability distribution over actions for a given state"""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.network(state_t)
            probs = torch.softmax(logits, dim=-1)
        return probs.numpy()[0]


ACTION_TO_PHASE = {0: 0, 1: 2}

# Global variables for enhanced state tracking
prev_queue_lengths = None
time_since_last_switch = 0


def get_state():
    global prev_queue_lengths, time_since_last_switch
    
    state = []
    
    # Current queue state
    current_queues = []
    for lane in LANES:
        state.append(traci.lane.getLastStepHaltingNumber(lane))
        current_queues.append(traci.lane.getLastStepHaltingNumber(lane))
    
    # Queue lengths (occupancy)
    for lane in LANES:
        state.append(traci.lane.getLastStepLength(lane))
    
    # Waiting times
    for lane in LANES:
        state.append(traci.lane.getWaitingTime(lane))
    
    # Rate of change in queue lengths (helps predict trends)
    if prev_queue_lengths is not None:
        for i in range(len(LANES)):
            state.append(current_queues[i] - prev_queue_lengths[i])
    else:
        state.extend([0] * len(LANES))
    
    prev_queue_lengths = current_queues
    
    # Current phase and time since last switch
    state.append(float(traci.trafficlight.getPhase(TL_ID)))
    state.append(min(time_since_last_switch, 100) / 100.0)  # Normalized
    
    return state


def get_wait_time():
    return sum(traci.lane.getWaitingTime(lane) for lane in LANES)


def run_baseline():
    global prev_queue_lengths, time_since_last_switch
    prev_queue_lengths = None
    time_since_last_switch = 0
    
    traci.start(["sumo", "-c", CONFIG_PATH])
    step = 0
    total_wait = 0.0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        if step % 200 == 0:
            phase = traci.trafficlight.getPhase(TL_ID)
            traci.trafficlight.setPhase(TL_ID, 2 if phase == 0 else 0)
        total_wait += get_wait_time()
        step += 1
    traci.close()
    return total_wait


def run_webster():
    global prev_queue_lengths, time_since_last_switch
    prev_queue_lengths = None
    time_since_last_switch = 0
    
    MAX_STEP = 3600
    traci.start(["sumo", "-c", CONFIG_PATH])
    step = 0

    edges = set(traci.edge.getIDList())
    tls = traci.trafficlight.getIDList()

    L = 8  # lost time because 2*4
    fixed_flow = 1850

    north_flow = []
    south_flow = []
    east_flow = []
    west_flow = []
    nf_count = []
    sf_count = []
    ef_count = []
    wf_count = []

    nextCycle = []
    for tl in tls:
        north_flow.append(0)
        south_flow.append(0)
        east_flow.append(0)
        west_flow.append(0)
        nf_count.append(set())
        sf_count.append(set())
        ef_count.append(set())
        wf_count.append(set())
        nextCycle.append(42)
    # total_wait
    sumWait = 0
    total_vehs = 0

    while traci.simulation.getMinExpectedNumber() > 0 and step <= MAX_STEP:
        step += 1
        for tl in range(len(tls)):
            GNS = 0
            GEW = 0
            flowLanes = sorted(set(traci.trafficlight.getControlledLanes(tls[tl])))

            if step == nextCycle[tl]:
                east_flow[tl] = len(ef_count[tl]) * MAX_STEP / step
                north_flow[tl] = len(nf_count[tl]) * MAX_STEP / step
                south_flow[tl] = len(sf_count[tl]) * MAX_STEP / step
                west_flow[tl] = len(wf_count[tl]) * MAX_STEP / step

                sat_north = north_flow[tl] / fixed_flow
                sat_south = south_flow[tl] / fixed_flow
                sat_east = east_flow[tl] / fixed_flow
                sat_west = west_flow[tl] / fixed_flow

                maxOfNS = max(sat_north, sat_south, 0.001)
                maxOfEW = max(sat_east, sat_west, 0.001)
                sumOfAllSats = maxOfNS + maxOfEW
                d = min(42, (1.5 * L + 5) / (1 - sumOfAllSats))
                GNS += (maxOfNS / sumOfAllSats) * (d - L)
                GEW += (maxOfEW / sumOfAllSats) * (d - L)
                nextCycle[tl] += math.ceil(d)

                for e in edges:
                    edgeOcc = traci.edge.getLastStepOccupancy(e)
                    if edgeOcc >= 0.9:
                        if e[0] == 'n' or e[0] == 's':
                            if e[0] == 'e' or e[0] == 'w':
                                continue
                            traci.trafficlight.setPhase(tls[tl], 0)
                        else:
                            traci.trafficlight.setPhase(tls[tl], 2)

                traci.trafficlight.setPhase(tls[tl], 2)
                traci.trafficlight.setPhaseDuration(tls[tl], math.ceil(GEW))
                traci.trafficlight.setPhase(tls[tl], 0)
                traci.trafficlight.setPhaseDuration(tls[tl], math.ceil(GNS))

            ef_count[tl] = ef_count[tl].union(traci.lane.getLastStepVehicleIDs(flowLanes[0]))
            nf_count[tl] = nf_count[tl].union(traci.lane.getLastStepVehicleIDs(flowLanes[1]))
            sf_count[tl] = sf_count[tl].union(traci.lane.getLastStepVehicleIDs(flowLanes[2]))
            wf_count[tl] = wf_count[tl].union(traci.lane.getLastStepVehicleIDs(flowLanes[3]))

        for e in edges:
            vehsNow = traci.edge.getLastStepVehicleIDs(e)
            for id in vehsNow:
                sumWait += traci.vehicle.getWaitingTime(id)
                total_vehs += 1

        traci.simulationStep()

    traci.close()
    return sumWait


def run_episode(agent, training=True):
    global prev_queue_lengths, time_since_last_switch
    
    # Reset global state tracking
    prev_queue_lengths = None
    time_since_last_switch = 0
    
    traci.start(["sumo", "-c", CONFIG_PATH])

    step = 0
    total_wait = 0.0
    yellow_countdown = 0
    pending_phase = None
    current_green = 0
    accumulated_reward = 0.0
    switch_count = 0
    total_throughput = 0

    prev_state = None
    prev_action = None
    prev_log_prob = None
    prev_value = None

    while traci.simulation.getMinExpectedNumber() > 0:
        time_since_last_switch += 1
        
        if yellow_countdown > 0:
            yellow_countdown -= 1
            if yellow_countdown == 0 and pending_phase is not None:
                traci.trafficlight.setPhase(TL_ID, pending_phase)
                current_green = pending_phase
                pending_phase = None
                switch_count += 1
                time_since_last_switch = 0  # Reset counter on switch
        elif step % DECISION_INTERVAL == 0:
            state = get_state()

            if training and prev_state is not None:
                agent.store(prev_state, prev_action, prev_log_prob,
                            accumulated_reward, prev_value, False)
                accumulated_reward = 0.0

            action, log_prob, value = agent.select_action(state)
            desired_phase = ACTION_TO_PHASE[action]

            if desired_phase != current_green:
                yellow_phase = 1 if current_green == 0 else 3
                traci.trafficlight.setPhase(TL_ID, yellow_phase)
                yellow_countdown = YELLOW_DURATION
                pending_phase = desired_phase
            else:
                traci.trafficlight.setPhase(TL_ID, current_green)

            prev_state = state
            prev_action = action
            prev_log_prob = log_prob
            prev_value = value

        traci.simulationStep()
        wait = get_wait_time()
        total_wait += wait
        
        # Improved reward function
        # 1. Penalize waiting (primary objective)
        wait_penalty = -wait
        
        # 2. Small penalty for switches (yellow lights waste time)
        switch_penalty = -10 if yellow_countdown == YELLOW_DURATION else 0
        
        # 3. Reward for vehicles that complete their journey
        arrived = len(traci.simulation.getArrivedIDList())
        throughput_reward = arrived * 5
        total_throughput += arrived
        
        # 4. Penalize queue imbalance (prevent one direction from starving)
        queue_lengths = [traci.lane.getLastStepHaltingNumber(lane) for lane in LANES]
        max_queue = max(queue_lengths)
        min_queue = min(queue_lengths)
        imbalance_penalty = -(max_queue - min_queue) * 0.5
        
        accumulated_reward += wait_penalty + switch_penalty + throughput_reward + imbalance_penalty
        step += 1

    if training and prev_state is not None:
        agent.store(prev_state, prev_action, prev_log_prob,
                    accumulated_reward, prev_value, True)
        agent.update()
        agent.scheduler.step()

    traci.close()
    return total_wait


# Code pulled from: https://github.com/AArdaNalbant/Traffic-Signal-Modification-with-Webster-Method/blob/master/traffic_light_management_system/runner.py#L168
# Modified to function soley for inference
def analyze_policy_confidence(agent):
    """
    Analyze how confident/deterministic the agent's policy is.
    A well-learned policy should have low entropy (confident decisions).
    """
    test_states = []
    
    # Generate various traffic scenarios
    for i in range(20):
        # Random but realistic traffic states
        queues = np.random.randint(0, 15, 4)
        lengths = queues * 5
        waits = queues * 10
        state = list(queues) + list(lengths) + list(waits) + [0]*4 + [float(i % 2 * 2), i/20.0]
        test_states.append(state)
    
    # Check action distribution across states
    action_probs = []
    for state in test_states:
        probs = agent.get_action_distribution(state)
        action_probs.append(probs)
    
    action_probs = np.array(action_probs)
    avg_entropy = -np.mean(np.sum(action_probs * np.log(action_probs + 1e-8), axis=1))
    max_prob = np.mean(np.max(action_probs, axis=1))
    
    print(f"\nPolicy Confidence Analysis (20 test scenarios):")
    print(f"  Average entropy: {avg_entropy:.4f} (lower = more confident)")
    print(f"  Average max probability: {max_prob:.3f} (higher = more decisive)")
    print(f"  Interpretation: {'✅ Confident policy' if max_prob > 0.7 else '⚠️  Uncertain policy'}")
    
    return max_prob > 0.7


def verify_learning(agent):
    """
    Test if agent has learned meaningful patterns vs random behavior.
    Checks if agent makes consistent, logical decisions on test scenarios.
    """
    print("\n" + "="*60)
    print("LEARNING VERIFICATION TEST")
    print("="*60)
    
    # Test scenario 1: Heavy North-South traffic
    state_heavy_ns = [
        10, 0, 0, 0,  # Halting: 10 cars north, 0 everywhere else
        50, 0, 0, 0,  # Queue length
        100, 0, 0, 0,  # Wait time
        0, 0, 0, 0,   # Queue change
        2.0, 0.5      # Phase 2 (EW green), long time since switch
    ]
    probs_heavy_ns = agent.get_action_distribution(state_heavy_ns)
    
    # Test scenario 2: Heavy East-West traffic
    state_heavy_ew = [
        0, 0, 10, 0,  # Halting: 10 cars east, 0 everywhere else
        0, 0, 50, 0,  # Queue length
        0, 0, 100, 0, # Wait time
        0, 0, 0, 0,   # Queue change
        0.0, 0.5      # Phase 0 (NS green), long time since switch
    ]
    probs_heavy_ew = agent.get_action_distribution(state_heavy_ew)
    
    # Test scenario 3: Balanced traffic
    state_balanced = [
        5, 5, 5, 5,   # Equal traffic
        25, 25, 25, 25,
        50, 50, 50, 50,
        0, 0, 0, 0,
        0.0, 0.2      # NS green, recent switch
    ]
    probs_balanced = agent.get_action_distribution(state_balanced)
    
    print("\nScenario 1: Heavy NORTH-SOUTH traffic, but light is E-W green")
    print(f"  Action probabilities: NS green={probs_heavy_ns[0]:.3f}, EW green={probs_heavy_ns[1]:.3f}")
    print(f"  Expected: Should prefer NS green (action 0)")
    print(f"  Result: {'✅ LEARNED' if probs_heavy_ns[0] > 0.6 else '❌ NOT LEARNED'}")
    
    print("\nScenario 2: Heavy EAST-WEST traffic, but light is N-S green")
    print(f"  Action probabilities: NS green={probs_heavy_ew[0]:.3f}, EW green={probs_heavy_ew[1]:.3f}")
    print(f"  Expected: Should prefer EW green (action 1)")
    print(f"  Result: {'✅ LEARNED' if probs_heavy_ew[1] > 0.6 else '❌ NOT LEARNED'}")
    
    print("\nScenario 3: Balanced traffic, recently switched to NS")
    print(f"  Action probabilities: NS green={probs_balanced[0]:.3f}, EW green={probs_balanced[1]:.3f}")
    print(f"  Expected: Should keep current (avoid thrashing)")
    print(f"  Result: {'✅ LEARNED' if probs_balanced[0] > 0.55 else '⚠️  UNCERTAIN'}")
    
    # Calculate confidence score
    learned_count = 0
    if probs_heavy_ns[0] > 0.6: learned_count += 1
    if probs_heavy_ew[1] > 0.6: learned_count += 1
    if probs_balanced[0] > 0.55: learned_count += 1
    
    print(f"\n{'=' * 60}")
    print(f"Learning Confidence: {learned_count}/3 scenarios show learned behavior")
    if learned_count >= 2:
        print("✅ Agent appears to have LEARNED meaningful patterns")
    else:
        print("❌ Agent may be relying on LUCK rather than learning")
    print(f"{'=' * 60}\n")
    
    return learned_count >= 2


def plot_results(episode_improvements, improvement, benchmark: str):
    # Plot learning curve
    num_episodes_actual = len(episode_improvements)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_episodes_actual + 1), episode_improvements, alpha=0.3, label='Raw Episodes')

    # Add moving average to show trend
    window = min(10, num_episodes_actual // 5)
    if num_episodes_actual >= window:
        moving_avg_b = []
        moving_avg_w = []
        for i in range(len(episode_improvements)):
            start = max(0, i - window + 1)
            moving_avg_b.append(np.mean(episode_improvements[start:i+1]))
        plt.plot(range(1, num_episodes_actual + 1), moving_avg_b, linewidth=2, label=f'{window}-Episode Moving Avg')

    plt.axhline(y=30, color='r', linestyle='--', label='30% Target')
    plt.axhline(y=improvement, color='g', linestyle='--', label=f'{benchmark} Eval: {improvement:.1f}%')

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel(f'Improvement vs {benchmark} (%)', fontsize=12)
    plt.title(f'PPO Training Progress ({num_episodes_actual} episodes)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'training_progress_{benchmark}.png', dpi=150)
    print(f"Training graph saved to training_progress_{benchmark}.png")


def main():
    ## Random seed for reproducibility
    torch.manual_seed(15)
    np.random.seed(15)


    print("Running baseline (fixed 200-step toggle)...")
    baseline_wait = run_baseline()
    print(f"Baseline: {baseline_wait:.2f}s total | {baseline_wait / 520:.2f}s per car\n")

    print("Running Websters...")
    webster_wait = run_webster()
    print(f"Webster: {webster_wait:.2f}s total | {webster_wait / 520:.2f}s per car\n")
    
    agent = PPOAgent()
    episode_improvements_b = []
    episode_improvements_w = []
    
    # Early stopping variables
    best_wait = float('inf')
    patience = 25  # Increased from 15 to allow more exploration
    no_improvement_count = 0
    best_episode = 0
    
    # Track learning progress
    entropy_history = []
    value_loss_history = []

    print(f"Training PPO agent for {NUM_EPISODES} episodes...\n")
    print(f"{'Episode':<8} {'Wait Time':<12} {'vs Base':<10} {'vs Webster':<12} {'Policy Loss':<12} {'Value Loss':<12} {'Entropy':<10} {'Status'}")
    print("-" * 100)
    
    for ep in range(NUM_EPISODES):
        ep_wait = run_episode(agent, training=True)
        pct_b = ((baseline_wait - ep_wait) / baseline_wait) * 100
        pct_w = ((webster_wait - ep_wait) / webster_wait) * 100
        episode_improvements_b.append(pct_b)
        episode_improvements_w.append(pct_w)
        
        # Get learning diagnostics
        policy_loss = agent.last_policy_loss if hasattr(agent, 'last_policy_loss') and agent.last_policy_loss else 0
        value_loss = agent.last_value_loss if hasattr(agent, 'last_value_loss') and agent.last_value_loss else 0
        entropy = agent.last_entropy if hasattr(agent, 'last_entropy') and agent.last_entropy else 0
        
        # Track metrics for analysis
        entropy_history.append(entropy)
        value_loss_history.append(value_loss)

        # Save best model and check for improvement
        if ep_wait < best_wait:
            best_wait = ep_wait
            best_episode = ep + 1
            no_improvement_count = 0
            torch.save(agent.network.state_dict(), f"best_model_ep{ep+1}.pt")
            status = "✓ NEW BEST"
            print(f"{ep+1:<8} {ep_wait:>10.0f}s  {pct_b:>+6.1f}%   {pct_w:>+8.1f}%    "
                  f"{policy_loss:>9.4f}    {value_loss:>9.4f}     {entropy:>6.4f}    {status}")
        else:
            no_improvement_count += 1
            status = f"({no_improvement_count}/{patience})"
            print(f"{ep+1:<8} {ep_wait:>10.0f}s  {pct_b:>+6.1f}%   {pct_w:>+8.1f}%    "
                  f"{policy_loss:>9.4f}    {value_loss:>9.4f}     {entropy:>6.4f}    {status}")
        
        # Early stopping check
        if no_improvement_count >= patience:
            print(f"\n⚠️  Early stopping triggered after {ep + 1} episodes")
            print(f"   No improvement for {patience} consecutive episodes")
            print(f"   Best model was at episode {best_episode} with wait time: {best_wait:.2f}s")
            break
    
    # Analyze learning metrics
    print("\n" + "="*60)
    print("LEARNING ANALYSIS")
    print("="*60)
    
    if len(entropy_history) > 10:
        early_entropy = np.mean(entropy_history[:10])
        late_entropy = np.mean(entropy_history[-10:])
        entropy_change = ((early_entropy - late_entropy) / early_entropy) * 100
        
        print(f"Entropy (randomness in decisions):")
        print(f"  Early episodes (1-10):  {early_entropy:.4f}")
        print(f"  Late episodes:          {late_entropy:.4f}")
        print(f"  Change: {entropy_change:+.1f}% {'(✅ more confident)' if entropy_change > 10 else '(⚠️  still random)'}")
        
    if len(value_loss_history) > 10:
        early_vloss = np.mean(value_loss_history[:10])
        late_vloss = np.mean(value_loss_history[-10:])
        vloss_change = ((early_vloss - late_vloss) / early_vloss) * 100
        
        print(f"\nValue Loss (prediction accuracy):")
        print(f"  Early episodes:  {early_vloss:.4f}")
        print(f"  Late episodes:   {late_vloss:.4f}")
        print(f"  Change: {vloss_change:+.1f}% {'(✅ better predictions)' if vloss_change > 10 else '(⚠️  not improving)'}")
    
    print("="*60)


    print("\nFinal evaluation (loading best model)...")
    print(f"Loading best model from episode {best_episode}...")
    agent.network.load_state_dict(torch.load(f"best_model_ep{best_episode}.pt"))
    agent.network.eval()
    
    eval_wait = run_episode(agent, training=False)
    improvement_baseline = ((baseline_wait - eval_wait) / baseline_wait) * 100
    improvement_webster = ((webster_wait - eval_wait) / webster_wait) * 100

    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"  Baseline:    {baseline_wait:>12.2f}s  ({baseline_wait / 520:.2f}s/car)")
    print(f"  Webster :    {webster_wait:>12.2f}s  ({webster_wait / 520:.2f}s/car)")
    print(f"  PPO Agent:   {eval_wait:>12.2f}s  ({eval_wait / 520:.2f}s/car)")
    print(f"  Best Episode: {best_episode}")
    print(f"  Improvement over Baseline: {improvement_baseline:>+11.1f}%")
    print(f"  Improvement over Webster : {improvement_webster:>+11.1f}%")
    print(f"  Target:              30%")
    print(f"{'=' * 60}")

    torch.save(agent.network.state_dict(), "ppo_traffic_model.pt")
    print("\nBest model saved to ppo_traffic_model.pt")
    
    # Verify the agent actually learned patterns
    agent.network.load_state_dict(torch.load(f"best_model_ep{best_episode}.pt"))
    agent.network.eval()
    has_learned = verify_learning(agent)
    is_confident = analyze_policy_confidence(agent)
    
    if has_learned and is_confident:
        print("\n✅ CONCLUSION: Agent demonstrates genuine learning")
    elif has_learned or is_confident:
        print("\n⚠️  CONCLUSION: Agent shows some learning but may benefit from more training")
    else:
        print("\n❌ CONCLUSION: Agent may be relying on luck rather than learned patterns")
    
    plot_results(episode_improvements_b, improvement_baseline, "baseline")
    plot_results(episode_improvements_w, improvement_webster, "webster")


if __name__ == "__main__":
    main()
