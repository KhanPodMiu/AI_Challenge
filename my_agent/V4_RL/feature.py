import numpy as np


def encode_obs(obs, agent_id):

    tensor = np.zeros((9, 13, 13), dtype=np.float32)

    game_map = obs["map"]

    players = obs["players"]

    bombs = obs["bombs"]

    # channel 0 = walls
    tensor[0] = (game_map == 1)

    # channel 1 = boxes
    tensor[1] = (game_map == 2)

    # channel 2 = radius items
    tensor[2] = (game_map == 3)

    # channel 3 = capacity items
    tensor[3] = (game_map == 4)

    # channel 4 = bombs
    for bomb in bombs:

        r, c, timer, owner = bomb

        tensor[4, r, c] = 1.0

        # channel 5 = bomb timer
        tensor[5, r, c] = timer / 7.0

    # self
    my_r, my_c = players[agent_id][:2]

    tensor[6, my_r, my_c] = 1.0

    # enemies
    for i, player in enumerate(players):

        if i == agent_id:
            continue

        r, c, alive, _, _ = player

        if alive:
            tensor[7, r, c] = 1.0

    # walkable
    tensor[8] = (game_map == 0)

    return tensor