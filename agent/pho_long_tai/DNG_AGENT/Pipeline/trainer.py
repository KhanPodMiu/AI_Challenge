import torch
import torch.nn.functional as F


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        gamma=0.99
    ):

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

        states = torch.tensor(
            states,
            dtype=torch.float32
        )

        actions = torch.tensor(actions)

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32
        )

        next_states = torch.tensor(
            next_states,
            dtype=torch.float32
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32
        )

        # current Q
        q_values = self.model(states)

        current_q = q_values.gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        # next Q
        with torch.no_grad():

            next_q = self.model(next_states)

            max_next_q = next_q.max(1)[0]

        target_q = rewards + (
            1 - dones
        ) * self.gamma * max_next_q

        # loss
        loss = F.mse_loss(
            current_q,
            target_q
        )

        # update
        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return loss.item()