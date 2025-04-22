# train_rl.py — обучение агента через Deep Q-Learning

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# === Параметры обучения ===
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
REPLAY_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 5000
TARGET_UPDATE = 100
EPISODES = 500

# === Состояние: 5x5 тайлов + 1 флаг (внутри/снаружи) ===
STATE_SIZE = 25 + 1
ACTION_SIZE = 4

# === Q-Network ===
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_SIZE)
        )

    def forward(self, x):
        return self.net(x)

# === Вспомогательные ===
def get_epsilon(step):
    return EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-step / EPSILON_DECAY)

def preprocess_state(vision, inside):
    flat = np.array(sum(vision, []), dtype=np.float32)
    return np.append(flat, [1.0 if inside else 0.0])

# === Заглушка среды (будет заменена реальной) ===
class DummyEnv:
    def reset(self):
        return np.zeros((5, 5)), False

    def step(self, action):
        next_state = np.zeros((5, 5))
        reward = random.random()
        done = False
        inside = random.choice([True, False])
        return next_state, reward, done, inside

# === Главная тренировка ===
def train():
    env = DummyEnv()  # позже заменим на GameEnv
    q_net = QNet()
    target_net = QNet()
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    memory = deque(maxlen=REPLAY_SIZE)
    step_count = 0

    for ep in range(EPISODES):
        vision, inside = env.reset()
        state = preprocess_state(vision, inside)
        total_reward = 0
        done = False

        while not done:
            step_count += 1
            epsilon = get_epsilon(step_count)
            if random.random() < epsilon:
                action = random.randint(0, ACTION_SIZE - 1)
            else:
                with torch.no_grad():
                    q_values = q_net(torch.tensor(state).unsqueeze(0))
                    action = torch.argmax(q_values).item()

            next_vision, reward, done, next_inside = env.step(action)
            next_state = preprocess_state(next_vision, next_inside)

            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # Обучение
            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                states_tensor = torch.tensor(states)
                actions_tensor = torch.tensor(actions).unsqueeze(1)
                rewards_tensor = torch.tensor(rewards).float().unsqueeze(1)
                next_states_tensor = torch.tensor(next_states)
                dones_tensor = torch.tensor(dones).float().unsqueeze(1)

                q_vals = q_net(states_tensor).gather(1, actions_tensor)
                with torch.no_grad():
                    next_q_vals = target_net(next_states_tensor).max(1, keepdim=True)[0]
                    target = rewards_tensor + GAMMA * next_q_vals * (1 - dones_tensor)

                loss = nn.MSELoss()(q_vals, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if step_count % TARGET_UPDATE == 0:
                target_net.load_state_dict(q_net.state_dict())

        print(f"Episode {ep+1}, reward: {total_reward:.2f}, epsilon: {epsilon:.2f}")

    torch.save(q_net.state_dict(), "agent_model_rl.pt")
    print("[✓] RL-модель сохранена: agent_model_rl.pt")

if __name__ == "__main__":
    train()
