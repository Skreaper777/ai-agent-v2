import json
from enum import Enum, auto
from collections import deque
from typing import List, Tuple, Dict, Optional

import numpy as np
import pygame
import heapq

# -----------------------------
# 🚀  High‑level API the rest of the game can use
# -----------------------------
# Agent is *self‑contained*: give it a WorldAdapter providing get_tile() & in_bounds()
# plus a dict of named zones (name -> set[(x,y)]). It will explore, remember, and
# switch motivations between «EXPLORE», «GO_HOME» and «GO_OUTSIDE».


class Motivation(Enum):
    EXPLORE = auto()
    GO_HOME = auto()
    GO_OUTSIDE = auto()


class MapMemory:
    """Occupancy grid seen by the agent.
    -1 = unknown, 0 = wall, 1 = free
    """

    def __init__(self, size: Tuple[int, int]):
        self._grid = np.full(size, -1, dtype=np.int8)

    def update(self, world, pos: Tuple[int, int], vision_radius: int = 2):
        cx, cy = pos
        for dy in range(-vision_radius, vision_radius + 1):
            for dx in range(-vision_radius, vision_radius + 1):
                nx, ny = cx + dx, cy + dy
                if world.in_bounds((nx, ny)):
                    self._grid[ny, nx] = 0 if world.is_blocked((nx, ny)) else 1

    def __getitem__(self, xy: Tuple[int, int]):
        x, y = xy
        return self._grid[y, x]

    def frontier(self) -> List[Tuple[int, int]]:
        """Returns all unknown neighbours of known free cells."""
        h, w = self._grid.shape
        result = []
        for y in range(h):
            for x in range(w):
                if self._grid[y, x] == 1:
                    for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                        if 0 <= nx < w and 0 <= ny < h and self._grid[ny, nx] == -1:
                            result.append((nx, ny))
        return result

    def known_free(self) -> List[Tuple[int, int]]:
        ys, xs = np.where(self._grid == 1)
        return list(zip(xs, ys))


# ----------  Simple A* on memory ----------

def heuristic(a: Tuple[int, int], b: Tuple[int, int]):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(mem: MapMemory, start: Tuple[int, int], goal: Tuple[int, int]):
    open_set = [(0 + heuristic(start, goal), 0, start, None)]
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
    g_score = {start: 0}

    while open_set:
        _, g, current, parent = heapq.heappop(open_set)
        if current in came_from:
            continue  # already processed with better cost
        came_from[current] = parent
        if current == goal:
            # reconstruct
            path = [current]
            while came_from[path[-1]] is not None:
                path.append(came_from[path[-1]])
            path.reverse()
            return path
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = current[0] + dx, current[1] + dy
            if mem[(nx, ny)] != 1:
                continue
            tentative = g + 1
            if tentative < g_score.get((nx, ny), 1e9):
                g_score[(nx, ny)] = tentative
                f = tentative + heuristic((nx, ny), goal)
                heapq.heappush(open_set, (f, tentative, (nx, ny), current))
    return []  # no path


# ----------  Demonstration buffer ----------

class DemoBuffer:
    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def record(self, state, action):
        self.buffer.append({"state": state, "action": action})

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(list(self.buffer), f, ensure_ascii=False, indent=2)


class Agent:
    """Intelligent (ish) agent that explores & has basic home/outside motivation."""

    def __init__(self, start: Tuple[int, int], world, zones: Dict[str, List[Tuple[int, int]]]):
        self.pos = start
        self.world = world
        self.mem = MapMemory(world.size)
        self.zones = zones
        self.motivation = Motivation.EXPLORE
        self.steps_in_home = 0
        self.steps_outside = 0
        self.current_path: List[Tuple[int, int]] = []

    # ------------- high‑level update -------------

    def tick(self):
        """Call each frame to update agent."""
        self.mem.update(self.world, self.pos)
        self._update_counters()
        self._update_motivation()
        if not self.current_path:
            goal = self._choose_goal()
            if goal is not None:
                self.current_path = astar(self.mem, self.pos, goal)[1:]
        if self.current_path:
            self.pos = self.current_path.pop(0)

    # ------------- internals -------------

    def _update_counters(self):
        if self.pos in self.zones.get("home", []):
            self.steps_in_home += 1
            self.steps_outside = 0
        else:
            self.steps_outside += 1
            self.steps_in_home = 0

    def _update_motivation(self):
        if self.steps_in_home > 200:
            self.motivation = Motivation.GO_OUTSIDE
        elif self.steps_outside > 300:
            self.motivation = Motivation.GO_HOME
        elif not self.current_path:
            self.motivation = Motivation.EXPLORE

    def _choose_goal(self) -> Optional[Tuple[int, int]]:
        if self.motivation == Motivation.EXPLORE:
            frontier = self.mem.frontier()
            return min(frontier, key=lambda c: heuristic(self.pos, c)) if frontier else None
        elif self.motivation == Motivation.GO_HOME:
            return min(self.zones.get("home", []), key=lambda c: heuristic(self.pos, c))
        elif self.motivation == Motivation.GO_OUTSIDE:
            outside_frontier = [c for c in self.mem.frontier() if c not in self.zones.get("home", [])]
            if outside_frontier:
                return min(outside_frontier, key=lambda c: heuristic(self.pos, c))
        return None

    # ------------- rendering helpers -------------

    def draw_path(self, surface, tile_size):
        for x, y in self.current_path:
            rect = pygame.Rect(x * tile_size, y * tile_size, tile_size, tile_size)
            pygame.draw.rect(surface, (255, 0, 0), rect, 1)

    def draw_ui(self, surface, x, y):
        font = pygame.font.SysFont("consolas", 16)
        text = font.render(f"Motivation: {self.motivation.name}", True, (255, 255, 255))
        surface.blit(text, (x, y))
        # Simple progress bars
        bar_w, bar_h = 100, 10
        pygame.draw.rect(surface, (100, 100, 100), (x, y + 20, bar_w, bar_h))
        pygame.draw.rect(surface, (0, 200, 0), (x, y + 20, min(self.steps_in_home, bar_w), bar_h))
        pygame.draw.rect(surface, (0, 0, 200), (x, y + 35, bar_w, bar_h))
        pygame.draw.rect(surface, (0, 200, 200), (x, y + 35, min(self.steps_outside, bar_w), bar_h))


# -------------  Convenience wrapper for game integration -------------


def integrate_agent(game):
    """Example one‑liner the main file can call after loading map."""
    home_cells = game.map.get_zone("home")  # assuming helper exists
    game.agent = Agent(game.agent_start, game.map, {"home": home_cells})
    game.demo_buffer = DemoBuffer()

    def game_update_agent(dt):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_r]:
            # record Adam player's demo
            game.demo_buffer.record(game.player.get_state(), game.player.last_action())
        game.agent.tick()

    game.add_update_callback(game_update_agent)
