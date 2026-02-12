import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("Please set SUMO_HOME environment variable!")

import traci

CONFIG_PATH = os.path.join("simulations", "Easy_4_Way", "map.sumocfg")
TL_ID = "TCenter"
LANES = ['NI_0', 'SI_0', 'EI_0', 'WI_0']

STATE_DIM = 14
ACTION_DIM = 2
DECISION_INTERVAL = 10
YELLOW_DURATION = 3
NUM_EPISODES = 20
GAMMA = 0.99
LR = 3e-4
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
        state.append(traci.lane.getLastStepLength(lane))
    for lane in LANES:
        state.append(traci.lane.getWaitingTime(lane))
    state.append(float(traci.trafficlight.getPhase(TL_ID)))
    state.append(traci.simulation.getTime())
    return state


def get_wait_time():
    return sum(traci.lane.getWaitingTime(lane) for lane in LANES)


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


def run_episode(agent, training=True):
    traci.start(["sumo", "-c", CONFIG_PATH])

    step = 0
    total_wait = 0.0
    yellow_countdown = 0
    pending_phase = None
    current_green = 0
    accumulated_reward = 0.0

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
        accumulated_reward += -wait
        step += 1

    if training and prev_state is not None:
        agent.store(prev_state, prev_action, prev_log_prob,
                    accumulated_reward, prev_value, True)
        agent.update()

    traci.close()
    return total_wait


def main():
    print("Running baseline (fixed 200-step toggle)...")
    baseline_wait = run_baseline()
    print(f"Baseline: {baseline_wait:.2f}s total | {baseline_wait / 400:.2f}s per car\n")

    agent = PPOAgent()

    print(f"Training PPO agent for {NUM_EPISODES} episodes...\n")
    for ep in range(NUM_EPISODES):
        ep_wait = run_episode(agent, training=True)
        pct = ((baseline_wait - ep_wait) / baseline_wait) * 100
        print(f"  Episode {ep + 1:>2}/{NUM_EPISODES} | "
              f"Wait: {ep_wait:>10.2f}s | "
              f"Per car: {ep_wait / 400:>7.2f}s | "
              f"vs Baseline: {pct:>+6.1f}%")

    print("\nFinal evaluation (no training)...")
    eval_wait = run_episode(agent, training=False)
    improvement = ((baseline_wait - eval_wait) / baseline_wait) * 100

    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"  Baseline:    {baseline_wait:>12.2f}s  ({baseline_wait / 400:.2f}s/car)")
    print(f"  PPO Agent:   {eval_wait:>12.2f}s  ({eval_wait / 400:.2f}s/car)")
    print(f"  Improvement: {improvement:>+11.1f}%")
    print(f"  Target:              30%")
    print(f"{'=' * 60}")

    torch.save(agent.network.state_dict(), "ppo_traffic_model.pt")
    print("\nModel saved to ppo_traffic_model.pt")


if __name__ == "__main__":
    main()
