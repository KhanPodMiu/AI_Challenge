import random
import torch

import sys
import os

ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)

sys.path.insert(0, ROOT_DIR)

from model import CNNModel
from feature import encode_obs
from replay_buffer import ReplayBuffer
from trainer import Trainer

from engine.game import BomberEnv

from agent import (
    SimpleRuleAgent,
    SmarterRuleAgent,
    TacticalRuleAgent,
    RandomAgent,
    GeniusRuleAgent,
    BoxFarmerAgent
)


if torch.backends.mps.is_available():

    device = torch.device("mps")

elif torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")

print(f"🔥 device = {device}")


# LOCAL

SAVE_DIR = "./checkpoints"

MODELS_DIR = "./models"


# COLAB

# SAVE_DIR = "/content/drive/MyDrive/bomber_rl_shared/checkpoints"

# MODELS_DIR = "/content/drive/MyDrive/bomber_rl_shared/models"


os.makedirs(SAVE_DIR, exist_ok=True)

os.makedirs(MODELS_DIR, exist_ok=True)


CHECKPOINT_PATH = os.path.join(
    SAVE_DIR,
    "latest_checkpoint.pth"
)

BEST_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "best_model.pth"
)

FINAL_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "final_model.pth"
)


env = BomberEnv()


enemy_bots = [
    SimpleRuleAgent(1),
    BoxFarmerAgent(2),
    SimpleRuleAgent(3)
]


model = CNNModel().to(device)

model.train()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)


trainer = Trainer(
    model,
    optimizer,
    device=device
)


buffer = ReplayBuffer(
    capacity=50000
)


episodes = 50

batch_size = 64

epsilon = 1.0

epsilon_min = 0.05

epsilon_decay = 0.9995

max_steps = 500

start_episode = 0

best_reward = -999999


if os.path.exists(CHECKPOINT_PATH):

    print("📦 loading checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    epsilon = checkpoint["epsilon"]

    start_episode = checkpoint["episode"] + 1

    best_reward = checkpoint["best_reward"]

    print(
        f"✅ resumed from episode {start_episode}"
    )

else:

    print("🆕 starting new training")


for episode in range(
    start_episode,
    episodes
):

    obs = env.reset()

    done = False

    total_reward = 0.0

    loss = 0.0

    step = 0

    while not done and step < max_steps:

        step += 1

        state = encode_obs(obs, 0)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        with torch.no_grad():

            q_values = model(state_tensor)

        if random.random() < epsilon:

            my_action = random.randint(0, 5)

        else:

            my_action = torch.argmax(
                q_values,
                dim=1
            ).item()

        enemy_actions = []

        for bot in enemy_bots:

            try:

                if hasattr(bot, "act"):

                    enemy_action = bot.act(obs)

                elif hasattr(bot, "get_action"):

                    enemy_action = bot.get_action(obs)

                else:

                    enemy_action = random.randint(0, 5)

            except Exception as e:

                print(f"⚠️ enemy bot error: {e}")

                enemy_action = random.randint(0, 5)

            enemy_actions.append(enemy_action)

        actions = [my_action] + enemy_actions

        next_obs, rewards, terminated, truncated = env.step(actions)

        reward = rewards[0]

        done = terminated or truncated

        next_state = encode_obs(next_obs, 0)

        buffer.push(
            state,
            my_action,
            reward,
            next_state,
            done
        )

        total_reward += reward

        if len(buffer) >= batch_size:

            batch = buffer.sample(batch_size)

            loss = trainer.train_step(batch)

        obs = next_obs

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

    if torch.is_tensor(loss):

        loss_value = loss.item()

    else:

        loss_value = loss

    print(
        f"[EP {episode:05d}] | "
        f"Reward: {total_reward:8.2f} | "
        f"Epsilon: {epsilon:.3f} | "
        f"Loss: {loss_value:.4f} | "
        f"Steps: {step:03d} | "
        f"Buffer: {len(buffer)}"
    )

    if total_reward > best_reward:

        best_reward = total_reward

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print("🏆 best model updated")

    if episode > 0 and episode % 10 == 0:

        torch.save({

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "epsilon":
                epsilon,

            "episode":
                episode,

            "best_reward":
                best_reward,

        }, CHECKPOINT_PATH)

        print("💾 checkpoint saved")


torch.save(
    model.state_dict(),
    FINAL_MODEL_PATH
)

print("✅ final model saved")
