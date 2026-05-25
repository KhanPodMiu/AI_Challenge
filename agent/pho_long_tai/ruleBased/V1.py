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

    def bfs(self, grid, start, goal):
        pass

    def dfs(self, grid, start, goal):
        pass
    


    def act(self, obs):

    
        

        grid = obs["map"]
        players = obs["players"]
        bombs = obs["bombs"]

        me = players[self.agent_id]

        if me[2] != 1:
            return 0

        x, y, alive, bombs_left, power = me

        my_pos = (x, y)

        enemies = []

        for i, p in enumerate(players):

            if i == self.agent_id:
                continue

            if p[2] == 1:
                enemies.append( (p[0], p[1]) )

        bomb_tiles = self.get_danger_tiles(grid, bombs, 1 + power)

        # ESCAPE

        if my_pos in bomb_tiles:

            move = self.escape(grid, my_pos, bomb_tiles)

            if move is not None:
                return move

        # ATTACK

        if bombs_left > 0:

            for ex, ey in enemies:

                dist = abs(ex - x) + abs(ey - y)

                if dist <= power + 1:

                    if self.clear_path(grid, my_pos, (ex, ey)):

                        if self.can_escape_after_bomb(
                            grid,
                            my_pos,
                            bombs,
                            power + 1
                        ):
                            return 5

        # BREAK BOX

        if bombs_left > 0:

            if self.near_box(grid, my_pos):

                if self.can_escape_after_bomb(
                    grid,
                    my_pos,
                    bombs,
                    power + 1
                ):
                    return 5

        # MOVE TO ITEM

        item_tiles = self.find_items(grid)

        move = self.move_to_target(
            grid,
            my_pos,
            item_tiles,
            bomb_tiles
        )

        if move is not None:
            return move

        # RANDOM SAFE MOVE

        valid = []

        for a in [1, 2, 3, 4]:

            nx, ny = self.next_pos(my_pos, a)

            if not self.passable(grid, nx, ny):
                continue

            if (nx, ny) in bomb_tiles:
                continue

            valid.append(a)

        if valid:
            return random.choice(valid)

        return 0

    # HELPERS

    def next_pos(self, pos, action):

        dx, dy = self.MOVES[action]

        return pos[0] + dx, pos[1] + dy

    def in_bounds(self, grid, x, y):

        return (
            0 <= x < grid.shape[0]
            and
            0 <= y < grid.shape[1]
        )

    def passable(self, grid, x, y):

        return (
            self.in_bounds(grid, x, y)
            and grid[x, y] in [0, 3, 4]
        )

    # DANGER

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

    def get_danger_tiles(self, grid, bombs, radius):

        danger = set()

        for b in bombs:

            bx, by = b[0], b[1]

            danger |= self.get_blast(grid, bx, by, radius)

        return danger

    # ESCAPE

    def escape(self, grid, start, danger):

        q = deque([(start, None)])

        visited = {start}

        while q:

            pos, first_move = q.popleft()

            if pos not in danger and first_move is not None:
                return first_move

            for a in [1,2,3,4]:

                nx, ny = self.next_pos(pos, a)

                npos = (nx, ny)

                if npos in visited:
                    continue

                if not self.passable(grid, nx, ny):
                    continue

                visited.add(npos)

                q.append(
                    (
                        npos,
                        a if first_move is None else first_move
                    )
                )

        return None

    # ITEMS

    def find_items(self, grid):

        items = set()

        for x in range(grid.shape[0]):

            for y in range(grid.shape[1]):

                if grid[x, y] in [3, 4]:
                    items.add((x, y))

        return items

    # MOVE TO TARGET

    def move_to_target(self, grid, start, targets, danger):

        if not targets:
            return None

        q = deque([(start, None)])

        visited = {start}

        while q:

            pos, first_move = q.popleft()

            if pos in targets and first_move is not None:
                return first_move

            for a in [1,2,3,4]:

                nx, ny = self.next_pos(pos, a)

                npos = (nx, ny)

                if npos in visited:
                    continue

                if not self.passable(grid, nx, ny):
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

    # COMBAT

    def clear_path(self, grid, a, b):

        ax, ay = a
        bx, by = b

        if ax == bx:

            step = 1 if by > ay else -1

            for y in range(ay + step, by, step):

                if grid[ax, y] in [1, 2]:
                    return False

            return True

        if ay == by:

            step = 1 if bx > ax else -1

            for x in range(ax + step, bx, step):

                if grid[x, ay] in [1, 2]:
                    return False

            return True

        return False

    def near_box(self, grid, pos):

        x, y = pos

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:

            nx = x + dx
            ny = y + dy

            if self.in_bounds(grid, nx, ny):

                if grid[nx, ny] == 2:
                    return True

        return False

    # BOMB SAFETY

    def can_escape_after_bomb(self, grid, pos, bombs, radius):

        danger = self.get_danger_tiles(grid, bombs, radius)

        my_blast = self.get_blast(
            grid,
            pos[0],
            pos[1],
            radius
        )

        total_danger = danger | my_blast

        move = self.escape(
            grid,
            pos,
            total_danger
        )

        return move is not None
    
    def bfs(self, grid, start, goal):
        return 1