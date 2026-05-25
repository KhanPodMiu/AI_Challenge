import random
import torch

from model import CNNModel
from replay_buffer import ReplayBuffer
from trainer import Trainer


model = CNNModel()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

trainer = Trainer(
    model,
    optimizer
)

buffer = ReplayBuffer(
    capacity=100000
)

for step in range(100000):

    # FAKE DATA

    state = torch.randn(9, 13, 13).numpy()

    next_state = torch.randn(9, 13, 13).numpy()

    action = random.randint(0, 5)

    reward = random.random()

    done = random.choice([0, 1])

    buffer.push(
        state,
        action,
        reward,
        next_state,
        done
    )

    # train
    if len(buffer) > 64:

        batch = buffer.sample(64)

        loss = trainer.train_step(batch)

        if step % 100 == 0:

            print(
                f"step={step} loss={loss}"
            )