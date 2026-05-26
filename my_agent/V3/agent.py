from module1 import Helper

class Agent:
    def __init__(self, agent_id: int):
        # agent_id: 0, 1, 2, or 3
        self.agent_id = agent_id

    def act(self, obs: dict) -> int:
        # obs: dict containing 'map', 'players', 'bombs'
        # Returns: int in [0, 5]
        ...