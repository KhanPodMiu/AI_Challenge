import random
from collections import deque


class Agent:


    MOVES = {
        0: (0, 0),
        1: (-1, 0),
        2: (1, 0),
        3: (0, -1),
        4: (0, 1),
    }

    def __init__(self, agent_id):

        self.agent_id = int(agent_id)

        self.last_bomb_step = -999

    # =========================================================
    # MAIN
    # =========================================================

    def act(self, obs):

        grid = obs["map"]
        players = obs["players"]
        bombs = obs["bombs"]
        step = obs.get("step", 0)

        me = players[self.agent_id]

        if me[2] != 1:
            return 0

        x, y, alive, bombs_left, power = me

        my_pos = (int(x), int(y))

        bomb_radius = int(power) + 1

        enemies = []

        for i, p in enumerate(players):

            if i == self.agent_id:
                continue

            if p[2] == 1:

                enemies.append(
                    (
                        int(p[0]),
                        int(p[1])
                    )
                )

        danger = self.get_danger_tiles(
            grid,
            bombs,
            players
        )

        # =====================================================
        # ESCAPE
        # =====================================================

        if my_pos in danger:

            move = self.escape(
                grid,
                my_pos,
                danger
            )

            if move is not None:
                return move

        # =====================================================
        # BOMB ENEMY
        # =====================================================

        if bombs_left > 0:

            if step - self.last_bomb_step > 5:

                value = self.evaluate_bomb(
                    grid,
                    my_pos,
                    enemies,
                    bomb_radius
                )

                if value >= 50:

                    if self.can_escape_after_bomb(
                        grid,
                        my_pos,
                        bombs,
                        players,
                        bomb_radius
                    ):

                        self.last_bomb_step = step

                        return 5

        # =====================================================
        # ITEM HUNT
        # =====================================================

        item_tiles = self.find_items(grid)

        move = self.move_to_best_target(
            grid,
            my_pos,
            item_tiles,
            danger
        )

        if move is not None:
            return move

        # =====================================================
        # BOX FARMING
        # =====================================================

        farm_tiles = self.good_farm_positions(grid)

        move = self.move_to_best_target(
            grid,
            my_pos,
            farm_tiles,
            danger
        )

        if move is not None:
            return move

        # =====================================================
        # ENEMY PRESSURE
        # =====================================================

        move = self.chase_enemy(
            grid,
            my_pos,
            enemies,
            danger
        )

        if move is not None:
            return move

        # =====================================================
        # BEST SAFE MOVE
        # =====================================================

        return self.best_safe_move(
            grid,
            my_pos,
            enemies,
            danger
        )

    # =========================================================
    # POSITION HELPERS
    # =========================================================

    def next_pos(self, pos, action):

        dx, dy = self.MOVES[action]

        return (
            pos[0] + dx,
            pos[1] + dy
        )

    def in_bounds(self, grid, x, y):

        return (
            0 <= x < grid.shape[0]
            and
            0 <= y < grid.shape[1]
        )

    def passable(self, grid, x, y):

        return (
            self.in_bounds(grid, x, y)
            and
            grid[x, y] in [0, 3, 4]
        )

    # =========================================================
    # BLAST
    # =========================================================

    def get_blast(self, grid, bx, by, radius):

        tiles = {(bx, by)}

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:

            for r in range(1, radius + 1):

                x = bx + dx * r
                y = by + dy * r

                if not self.in_bounds(grid, x, y):
                    break

                if grid[x, y] == 1:
                    break

                tiles.add((x, y))

                if grid[x, y] == 2:
                    break

        return tiles

    def get_danger_tiles(self, grid, bombs, players):

        danger = set()

        for b in bombs:

            bx = int(b[0])
            by = int(b[1])

            owner = int(b[3]) if len(b) > 3 else -1

            radius = 1

            if 0 <= owner < len(players):

                radius = int(players[owner][4]) + 1

            danger |= self.get_blast(
                grid,
                bx,
                by,
                radius
            )

        return danger

    # =========================================================
    # ESCAPE
    # =========================================================

    def escape(self, grid, start, danger):

        q = deque([(start, None)])

        visited = {start}

        while q:

            pos, first_move = q.popleft()

            if pos not in danger and first_move is not None:
                return first_move

            for a in [1,2,3,4]:

                npos = self.next_pos(pos, a)

                if npos in visited:
                    continue

                if not self.passable(grid, npos[0], npos[1]):
                    continue

                visited.add(npos)

                q.append(
                    (
                        npos,
                        a if first_move is None else first_move
                    )
                )

        return None

    # =========================================================
    # ITEMS
    # =========================================================

    def find_items(self, grid):

        items = set()

        for x in range(grid.shape[0]):

            for y in range(grid.shape[1]):

                if grid[x, y] in [3,4]:

                    items.add((x, y))

        return items

    # =========================================================
    # FARMING
    # =========================================================

    def good_farm_positions(self, grid):

        spots = set()

        for x in range(grid.shape[0]):

            for y in range(grid.shape[1]):

                if grid[x, y] != 2:
                    continue

                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:

                    nx = x + dx
                    ny = y + dy

                    if self.passable(grid, nx, ny):

                        spots.add((nx, ny))

        return spots

    # =========================================================
    # PATHING
    # =========================================================

    def move_to_best_target(
        self,
        grid,
        start,
        targets,
        danger
    ):

        if not targets:
            return None

        q = deque([(start, None)])

        visited = {start}

        while q:

            pos, first_move = q.popleft()

            if pos in targets and first_move is not None:
                return first_move

            for a in [1,2,3,4]:

                npos = self.next_pos(pos, a)

                if npos in visited:
                    continue

                if not self.passable(grid, npos[0], npos[1]):
                    continue

                if npos in danger:
                    continue

                visited.add(npos)

                q.append(
                    (
                        npos,
                        a if first_move is None else first_move
                    )
                )

        return None

    # =========================================================
    # ENEMY CHASE
    # =========================================================

    def chase_enemy(
        self,
        grid,
        my_pos,
        enemies,
        danger
    ):

        if not enemies:
            return None

        best_move = None
        best_score = -999999

        center_x = grid.shape[0] // 2
        center_y = grid.shape[1] // 2

        for a in [1,2,3,4]:

            npos = self.next_pos(my_pos, a)

            if not self.passable(grid, npos[0], npos[1]):
                continue

            if npos in danger:
                continue

            score = 0

            # enemy pressure
            enemy_dist = min(
                abs(npos[0] - ex) + abs(npos[1] - ey)
                for ex, ey in enemies
            )

            score -= enemy_dist * 5

            # center control
            center_dist = (
                abs(npos[0] - center_x)
                +
                abs(npos[1] - center_y)
            )

            score -= center_dist

            # open space
            score += self.open_space_score(
                grid,
                npos
            )

            if score > best_score:

                best_score = score
                best_move = a

        return best_move

    # =========================================================
    # OPEN SPACE
    # =========================================================

    def open_space_score(self, grid, pos):

        score = 0

        for a in [1,2,3,4]:

            npos = self.next_pos(pos, a)

            if self.passable(
                grid,
                npos[0],
                npos[1]
            ):
                score += 1

        return score

    # =========================================================
    # BOMB EVALUATION
    # =========================================================

    def evaluate_bomb(
        self,
        grid,
        my_pos,
        enemies,
        radius
    ):

        score = 0

        blast = self.get_blast(
            grid,
            my_pos[0],
            my_pos[1],
            radius
        )

        # enemy hit
        for e in enemies:

            if e in blast:
                score += 100

        # crate value
        for x, y in blast:

            if grid[x, y] == 2:
                score += 15

        return score

    # =========================================================
    # ESCAPE AFTER BOMB
    # =========================================================

    def can_escape_after_bomb(
        self,
        grid,
        my_pos,
        bombs,
        players,
        radius
    ):

        future_danger = self.get_danger_tiles(
            grid,
            bombs,
            players
        )

        my_blast = self.get_blast(
            grid,
            my_pos[0],
            my_pos[1],
            radius
        )

        future_danger |= my_blast

        move = self.escape(
            grid,
            my_pos,
            future_danger
        )

        return move is not None

    # =========================================================
    # SAFE MOVE
    # =========================================================

    def best_safe_move(
        self,
        grid,
        my_pos,
        enemies,
        danger
    ):

        best_move = 0
        best_score = -999999

        center_x = grid.shape[0] // 2
        center_y = grid.shape[1] // 2

        for a in [0,1,2,3,4]:

            npos = self.next_pos(my_pos, a)

            if a != 0:

                if not self.passable(
                    grid,
                    npos[0],
                    npos[1]
                ):
                    continue

            score = 0

            # avoid danger
            if npos in danger:
                score -= 1000

            # center control
            center_dist = (
                abs(npos[0] - center_x)
                +
                abs(npos[1] - center_y)
            )

            score -= center_dist

            # open area
            score += self.open_space_score(
                grid,
                npos
            )

            # enemy pressure
            if enemies:

                enemy_dist = min(
                    abs(npos[0] - ex)
                    +
                    abs(npos[1] - ey)
                    for ex, ey in enemies
                )

                score -= enemy_dist * 2

            if score > best_score:

                best_score = score
                best_move = a

        return best_move