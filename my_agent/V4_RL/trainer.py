import torch
import torch.nn.functional as F
import numpy as np


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        device,
        gamma=0.99
    ):

        self.device = device

        self.model = model

        self.optimizer = optimizer

        self.gamma = gamma

    def train_step(
        self,
        batch
    ):

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for (
            state,
            action,
            reward,
            next_state,
            done
        ) in batch:

            states.append(state)

            actions.append(action)

            rewards.append(reward)

            next_states.append(next_state)

            dones.append(done)

        states = torch.from_numpy(
            np.array(states)
        ).float().to(self.device)

        actions = torch.tensor(
            actions,
            dtype=torch.long
        ).to(self.device)

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32
        ).to(self.device)

        next_states = torch.from_numpy(
            np.array(next_states)
        ).float().to(self.device)

        dones = torch.tensor(
            dones,
            dtype=torch.float32
        ).to(self.device)

        q_values = self.model(states)

        current_q = q_values.gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():

            next_q = self.model(next_states)

            max_next_q = next_q.max(1)[0]

        target_q = rewards + (
            1 - dones
        ) * self.gamma * max_next_q

        loss = F.mse_loss(
            current_q,
            target_q
        )

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return loss.item()
