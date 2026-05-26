import torch
import numpy as np

from model import CNNModel
from feature import encode_obs


class Agent:

    def __init__(self, agent_id):

        self.agent_id = agent_id

        self.model = CNNModel()

        self.model.load_state_dict(
            torch.load(
                "model.pth",
                map_location=torch.device("cpu")
            )
        )

        self.model.eval()

    def act(self, obs):

        state = encode_obs(obs, self.agent_id)

        state = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = self.model(state)

        action = torch.argmax(q_values).item()

        return action