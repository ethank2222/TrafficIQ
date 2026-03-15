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

STATE_DIM = 17
ACTION_DIM = 2
DECISION_INTERVAL = 10
YELLOW_DURATION = 3
NUM_EPISODES = 100
GAMMA = 0.99
LR = 1e-3
CLIP_EPS = 0.2
PPO_EPOCHS = 4
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
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
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

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

        returns = []
        R = 0
        for r, d in zip(reversed(self.rewards), reversed(self.dones)):
            R = r + GAMMA * R * (1 - d)
            returns.insert(0, R)

        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.stack(self.log_probs).detach()
        returns = torch.FloatTensor(returns)
        old_values = torch.stack(self.values).detach()

        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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

        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()


ACTION_TO_PHASE = {0: 0, 1: 2}


def get_state():
    state = []
    for lane in LANES:
        state.append(traci.lane.getLastStepHaltingNumber(lane))
    for lane in LANES:
        # replaced stepLength
        state.append(traci.lane.getLastStepMeanSpeed(lane))
    for lane in LANES:
        # replaced stepLength
        state.append(traci.lane.getLastStepOccupancy(lane))
    for lane in LANES:
        state.append(traci.lane.getWaitingTime(lane))
    state.append(float(traci.trafficlight.getPhase(TL_ID)))
    # state.append(traci.simulation.getTime())
    return state


def get_wait_time():
    return sum(traci.lane.getWaitingTime(lane) for lane in LANES)


def get_intermediate_reward(prev_wait):
    curr_wait = get_wait_time()
    delta_wait = prev_wait - curr_wait
    return -curr_wait + 0.5 * delta_wait, curr_wait


def run_baseline():
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


# Code pulled from: https://github.com/AArdaNalbant/Traffic-Signal-Modification-with-Webster-Method/blob/master/traffic_light_management_system/runner.py#L168
# Modified to function soley for inference
def run_webster():
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


def run_episode(agent, training=True, sumo_config="sumo"):
    traci.start([sumo_config, "-c", CONFIG_PATH])

    step = 0
    total_wait = 0.0
    yellow_countdown = 0
    pending_phase = None
    current_green = 0
    accumulated_reward = 0.0
    prev_wait = 0.0

    prev_state = None
    prev_action = None
    prev_log_prob = None
    prev_value = None

    while traci.simulation.getMinExpectedNumber() > 0:
        if yellow_countdown > 0:
            yellow_countdown -= 1
            if yellow_countdown == 0 and pending_phase is not None:
                traci.trafficlight.setPhase(TL_ID, pending_phase)
                current_green = pending_phase
                pending_phase = None
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

        # Intermediate reward: negative wait + improvement delta vs previous step
        step_reward, prev_wait = get_intermediate_reward(prev_wait)
        accumulated_reward += step_reward
        step += 1

    if training and prev_state is not None:
        agent.store(prev_state, prev_action, prev_log_prob,
                    accumulated_reward, prev_value, True)
        agent.update()

    traci.close()
    return total_wait


def plot_results(all_runs_improvements, all, best, worse, benchmark: str):
    plt.figure(figsize=(5, 6))
    
    all_runs = np.array(all_runs_improvements)  # shape: (10, NUM_EPISODES)
    episodes = range(1, NUM_EPISODES + 1)
    
    # Plot individual runs (slightly transparent)
    for run in all_runs:
        plt.plot(episodes, run, alpha=0.2, color='steelblue', linewidth=1)
    
    # Plot average across all runs
    avg_performance = np.mean(all_runs, axis=0)
    plt.plot(episodes, avg_performance, color='steelblue', linewidth=2.5, label='Average (10 runs)')

    plt.axhline(y=30, color='r', linestyle='--', label='30% Target')

    avg_improvement = np.mean(all)
    min_improvement = np.min(worse)
    max_improvement = np.max(best)

    plt.axhline(y=avg_improvement, color='g', linestyle='--', label=f'{benchmark} Avg Eval: {avg_improvement:.1f}%')
    plt.axhline(y=min_improvement, color='orange', linestyle=':', linewidth=1.5, label=f'{benchmark} Min Eval: {min_improvement:.1f}%')
    plt.axhline(y=max_improvement, color='purple', linestyle=':', linewidth=1.5, label=f'{benchmark} Max Eval: {max_improvement:.1f}%')

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel(f'Improvement vs {benchmark} (%)', fontsize=12)
    plt.title('PPO Training Progress', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'training_progress_{benchmark}.png', dpi=150)
    print(f"Training graph saved to training_progress_{benchmark}.png")


def main():
    print("Running baseline (fixed 200-step toggle)...")
    baseline_wait = run_baseline()
    print(f"Baseline: {baseline_wait:.2f}s total | {baseline_wait / 520:.2f}s per car\n")

    # print("Running Websters...")
    # webster_wait = run_webster()
    # print(f"Webster: {webster_wait:.2f}s total | {webster_wait / 520:.2f}s per car\n")
    
    agent = PPOAgent()
    all_episode_improvements_b = []
    # all_episode_improvements_w = []
    all_final = []
    best_peformance = -200.
    worst_peformance = 200.
    for i in range(5):
        print(f"FULL RUN: {i}")
        print(f"Training PPO agent for {NUM_EPISODES} episodes...\n")
        episode_improvements_b = []
        # episode_improvements_w = []
        for ep in range(NUM_EPISODES):
            ep_wait = run_episode(agent, training=True)
            pct_b = ((baseline_wait - ep_wait) / baseline_wait) * 100
            # pct_w = ((webster_wait - ep_wait) / webster_wait) * 100
            episode_improvements_b.append(pct_b)
            # episode_improvements_w.append(pct_w)
            print(f"  Episode {ep + 1:>2}/{NUM_EPISODES} | "
                f"Wait: {ep_wait:>10.2f}s | "
                f"Per car: {ep_wait / 520:>7.2f}s | "
                f"vs Baseline: {pct_b:>+6.1f}% | ")
                # f"vs Webster : {pct_w:>+6.1f}%")


        print("\nFinal evaluation (no training)...")
        eval_wait = run_episode(agent, training=False)
        improvement_baseline = ((baseline_wait - eval_wait) / baseline_wait) * 100
        # improvement_webster = ((webster_wait - eval_wait) / webster_wait) * 100

        print(f"\n{'=' * 60}")
        print(f"RESULTS")
        print(f"  Baseline:    {baseline_wait:>12.2f}s  ({baseline_wait / 520:.2f}s/car)")
        # print(f"  Webster :    {webster_wait:>12.2f}s  ({webster_wait / 520:.2f}s/car)")
        print(f"  PPO Agent:   {eval_wait:>12.2f}s  ({eval_wait / 520:.2f}s/car)")
        print(f"  Improvement over Baseline: {improvement_baseline:>+11.1f}%")
        # print(f"  Improvement over Webster : {improvement_webster:>+11.1f}%")
        print(f"  Target:              30%")
        print(f"{'=' * 60}")
        all_final.append(improvement_baseline)

        if improvement_baseline > best_peformance:
            best_peformance = improvement_baseline
            print(f"Iteration {i} best.")
            torch.save(agent.network.state_dict(), "ppo_traffic_model.pt")
            print("\nModel saved to ppo_traffic_model.pt")
        # kept as seperate if just to cover base case of first iteration being worst
        if improvement_baseline < worst_peformance:
            worst_peformance = improvement_baseline
            print(f"Iteration {i} worse.")

        all_episode_improvements_b.append(episode_improvements_b)
        # all_episode_improvements_w.append(episode_improvements_w)    
    plot_results(all_episode_improvements_b, all_final, best_peformance, worst_peformance, "baseline")
    # plot_results(all_episode_improvements_w, improvement_webster, "webster")


if __name__ == "__main__":
    main()
