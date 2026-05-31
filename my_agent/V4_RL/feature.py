import numpy as np

def build_danger_map(
    game_map,
    bombs,
    players
):

    danger = np.zeros(
        (13, 13),
        dtype=np.float32
    )

    dirs = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1)
    ]

    for bomb in bombs:

        r, c, timer, owner = bomb

        value = (7 - timer) / 7.0

        radius = 1 + players[owner][4]

        danger[r, c] = max(
            danger[r, c],
            value
        )

        for dr, dc in dirs:

            for step in range(1, radius + 1):

                nr = r + dr * step
                nc = c + dc * step

                if not (
                    0 <= nr < 13 and
                    0 <= nc < 13
                ):
                    break

                # wall
                if game_map[nr, nc] == 1:
                    break

                danger[nr, nc] = max(
                    danger[nr, nc],
                    value
                )

                # box
                if game_map[nr, nc] == 2:
                    break

    return danger

def encode_obs(obs, agent_id):

    #increase dimension
    tensor = np.zeros((10, 13, 13), dtype=np.float32)

    game_map = obs["map"]

    players = obs["players"]

    bombs = obs["bombs"]

    danger = build_danger_map(
        game_map,
        bombs,
        players
    )

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

    tensor[9] = danger

    return tensor