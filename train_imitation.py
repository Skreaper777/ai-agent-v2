# train_imitation.py — обучение агента по демонстрациям

import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

STATE_SIZE = 25  # 5x5 видимость (0 или 1)
ACTION_MAP = {"left": 0, "right": 1, "up": 2, "down": 3}
MODEL_PATH = "agent_model.pt"

# === Модель ===
class AgentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.net(x)

# === Загрузка буфера ===
def load_demo(path="demo_buffer.json"):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    X, y = [], []
    for item in data:
        state = np.array(sum(item["state"], []), dtype=np.float32)  # 5x5
        action = ACTION_MAP.get(item["action"])
        if action is not None:
            X.append(state)
            y.append(action)
    return np.array(X), np.array(y)

# === Обучение ===
def train():
    X, y = load_demo()
    if len(X) == 0:
        print("[!] Пустой demo_buffer — нечего обучать")
        return

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = AgentNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val)

    EPOCHS = 30
    print(f"[INFO] Обучающих примеров: {len(X)}")
    print(f"[INFO] Уникальных действий: {set(y)}")

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = model(X_train_t)
        loss = loss_fn(out, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t)
            val_loss = loss_fn(val_out, y_val_t).item()
            acc = (val_out.argmax(1) == y_val_t).float().mean().item()

        print(f"[{epoch+1:02d}] loss: {loss.item():.4f}, val_loss: {val_loss:.4f}, acc: {acc:.3f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[✓] Модель сохранена: {MODEL_PATH}")

if __name__ == "__main__":
    train()