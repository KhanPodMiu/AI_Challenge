import random

class Agent:
    def __init__(self, agent_id):
        self.agent_id = agent_id

    def act(self, obs):
        return random.randint(0, 5)