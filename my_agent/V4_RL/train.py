import random
import torch

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../.."
        )
    )
)

from model import CNNModel
from feature import encode_obs
from replay_buffer import ReplayBuffer
from trainer import Trainer

from engine.game import BomberEnv


# =========================================================
# DEVICE
# =========================================================

if torch.backends.mps.is_available():

    device = torch.device("mps")

elif torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")

print(f"🔥 device = {device}")


# =========================================================
# ENVIRONMENT
# =========================================================

env = BomberEnv()


# =========================================================
# MODEL
# =========================================================

model = CNNModel().to(device)


# =========================================================
# LOAD MODEL
# =========================================================

if os.path.exists("model.pth"):

    model.load_state_dict(
        torch.load(
            "model.pth",
            map_location=device,
            weights_only=False
        )
    )

    print("✅ old model loaded")


model.train()


# =========================================================
# OPTIMIZER
# =========================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)


# =========================================================
# TRAINER
# =========================================================

trainer = Trainer(
    model,
    optimizer,
    device=device
)


# =========================================================
# REPLAY BUFFER
# =========================================================

buffer = ReplayBuffer(
    capacity=50000
)


# =========================================================
# HYPERPARAMETERS
# =========================================================

episodes = 10000

batch_size = 64

gamma = 0.99

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995

max_steps = 300


# =========================================================
# TRAINING LOOP
# =========================================================

for episode in range(episodes):

    # reset env
    obs = env.reset()

    done = False

    total_reward = 0.0

    loss = 0.0

    step = 0

    while not done and step < max_steps:

        step += 1

        # =================================================
        # ENCODE STATE
        # =================================================

        state = encode_obs(obs, 0)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        # =================================================
        # GET ACTION
        # =================================================

        with torch.no_grad():

            q_values = model(state_tensor)

        # epsilon greedy
        if random.random() < epsilon:

            action = random.randint(0, 5)

        else:

            action = torch.argmax(
                q_values,
                dim=1
            ).item()

        # =================================================
        # OTHER PLAYERS
        # =================================================

        actions = [
            action,
            random.randint(0, 5),
            random.randint(0, 5),
            random.randint(0, 5)
        ]

        # =================================================
        # STEP ENV
        # =================================================

        next_obs, rewards, terminated, truncated = env.step(actions)

        reward = rewards[0]

        done = terminated or truncated

        # =================================================
        # NEXT STATE
        # =================================================

        next_state = encode_obs(next_obs, 0)

        # =================================================
        # SAVE TO BUFFER
        # =================================================

        buffer.push(
            state,
            action,
            reward,
            next_state,
            done
        )

        total_reward += reward

        # =================================================
        # TRAIN
        # =================================================

        if len(buffer) >= batch_size:

            batch = buffer.sample(batch_size)

            loss = trainer.train_step(batch)

        # =================================================
        # UPDATE OBS
        # =================================================

        obs = next_obs

    # =====================================================
    # EPSILON DECAY
    # =====================================================

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

    # =====================================================
    # LOSS VALUE
    # =====================================================

    if torch.is_tensor(loss):

        loss_value = loss.item()

    else:

        loss_value = loss

    # =====================================================
    # LOG
    # =====================================================

    print(
        f"[EP {episode:05d}] | "
        f"Reward: {total_reward:8.2f} | "
        f"Epsilon: {epsilon:.3f} | "
        f"Loss: {loss_value:.4f} | "
        f"Steps: {step:03d} | "
        f"Buffer: {len(buffer)}"
    )

    # =====================================================
    # SAVE MODEL
    # =====================================================

    if episode % 100 == 0:

        torch.save(
            model.state_dict(),
            "model.pth"
        )

        print("💾 checkpoint saved")


# =========================================================
# FINAL SAVE
# =========================================================

torch.save(
    model.state_dict(),
    "final_model.pth"
)

print("✅ final model saved")