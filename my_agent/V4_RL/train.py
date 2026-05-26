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


# device
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"device = {device}")


# env
env = BomberEnv()


# model
model = CNNModel().to(device)

# load old model
if os.path.exists("model.pth"):

    model.load_state_dict(
        torch.load(
            "model.pth",
            map_location=device
        )
    )

    print("model loaded")


model.train()


# optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)


# trainer
trainer = Trainer(
    model,
    optimizer,
    device=device
)


# replay buffer
buffer = ReplayBuffer(
    capacity=100000
)


# exploration
epsilon = 0.2


# training
for episode in range(300):

    obs = env.reset()

    done = False

    total_reward = 0

    loss = 0.0

    while not done:

        # encode state
        state = encode_obs(obs, 0)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        # predict q-values
        with torch.no_grad():

            q_values = model(state_tensor)

        # epsilon-greedy
        if random.random() < epsilon:

            action = random.randint(0, 5)

        else:

            action = torch.argmax(
                q_values
            ).item()

        # other bots random
        actions = [
            action,
            random.randint(0, 5),
            random.randint(0, 5),
            random.randint(0, 5)
        ]

        # game step
        next_obs, rewards, terminated, truncated = env.step(actions)

        reward = rewards[0]

        done = terminated or truncated

        # encode next state
        next_state = encode_obs(next_obs, 0)

        # save memory
        buffer.push(
            state,
            action,
            reward,
            next_state,
            done
        )

        total_reward += reward

        # train
        if len(buffer) > 64:

            batch = buffer.sample(64)

            loss = trainer.train_step(batch)

        # next obs
        obs = next_obs

    # epsilon decay
    epsilon = max(
        0.05,
        epsilon * 0.9995
    )

    # print training info
    print(
        f"[EP {episode:05d}] "
        f"Reward: {total_reward:8.2f} | "
        f"Epsilon: {epsilon:.3f} | "
        f"Loss: {loss:.4f} | "
        f"Buffer: {len(buffer)}"
    )

    # save model
    if episode % 100 == 0:

        torch.save(
            model.state_dict(),
            "model.pth"
        )

        print("🔥 model saved 🔥")