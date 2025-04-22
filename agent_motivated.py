import json
import heapq
import random
from enum import Enum, auto
from collections import deque
from typing import List, Tuple, Dict, Optional

import numpy as np
import pygame

Pos = Tuple[int, int]


class Motivation(Enum):
    EXPLORE = auto()
    GO_HOME = auto()
    GO_OUTSIDE = auto()


class MapMemory:
    """Матрица видимости: -1 неизвестно, 0 стена, 1 свободно."""

    def __init__(self, size: Tuple[int, int]):
        self.w, self.h = size
        self.grid = np.full((self.h, self.w), -1, dtype=np.int8)

    def update(self, world, pos: Pos, radius: int = 2) -> None:
        cx, cy = pos
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = cx + dx, cy + dy
                if world.in_bounds((nx, ny)):
                    self.grid[ny, nx] = 0 if world.is_blocked((nx, ny)) else 1

    def __getitem__(self, xy: Pos) -> int:
        x, y = xy
        if 0 <= x < self.w and 0 <= y < self.h:
            return self.grid[y, x]
        return 0  # за картой считаем стеной

    def frontier(self) -> List[Pos]:
        """Свободные клетки, у которых есть неизвестный сосед."""
        out: List[Pos] = []
        for y in range(self.h):
            for x in range(self.w):
                if self.grid[y, x] != 1:
                    continue
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if 0 <= nx < self.w and 0 <= ny < self.h and self.grid[ny, nx] == -1:
                        out.append((x, y))
                        break
        return out


# ---------- A* ---------- #
def _h(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _astar(mem: MapMemory, start: Pos, goal: Pos) -> List[Pos]:
    open_set = [(0 + _h(start, goal), 0, start, None)]
    came_from: Dict[Pos, Optional[Pos]] = {}
    g_score = {start: 0}

    while open_set:
        _, g, cur, parent = heapq.heappop(open_set)
        if cur in came_from:
            continue
        came_from[cur] = parent
        if cur == goal:
            path = [cur]
            while came_from[path[-1]] is not None:
                path.append(came_from[path[-1]])
            path.reverse()
            return path
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cur[0] + dx, cur[1] + dy
            if mem[(nx, ny)] != 1:
                continue
            tentative = g + 1
            if tentative < g_score.get((nx, ny), 1e9):
                g_score[(nx, ny)] = tentative
                heapq.heappush(open_set, (tentative + _h((nx, ny), goal), tentative, (nx, ny), cur))
    return []  # пути нет


# ---------- Демобуфер ---------- #
class DemoBuffer:
    def __init__(self, cap: int = 10_000):
        self.buf = deque(maxlen=cap)

    def record(self, st, act): self.buf.append({"state": st, "action": act})
    def save(self, p): json.dump(list(self.buf), open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


# ---------- Агент ---------- #
class Agent:
    def __init__(self, start: Pos, world, zones: Dict[str, List[Pos]]):
        self.pos = start
        self.world = world
        self.mem = MapMemory(world.size)
        self.zones = zones

        self.motivation = Motivation.EXPLORE
        self.steps_in_home = 0
        self.steps_outside = 0
        self.path: List[Pos] = []

    # -- public --
    def tick(self):
        self.mem.update(self.world, self.pos)
        self._count_steps()
        self._update_motivation()

        if not self.path:
            goal = self._choose_goal()
            if goal is not None:
                p = _astar(self.mem, self.pos, goal)
                self.path = p[1:] if len(p) > 1 else []

        # fallback — случайный свободный сосед, если так и не получилось
        if not self.path:
            neigh = [(self.pos[0] + dx, self.pos[1] + dy)
                     for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                     if self.mem[(self.pos[0]+dx, self.pos[1]+dy)] == 1]
            if neigh:
                self.path = [random.choice(neigh)]

        if self.path:
            self.pos = self.path.pop(0)

    def draw_path(self, surf, ts):
        for x, y in self.path:
            pygame.draw.rect(surf, (255, 0, 0), (x*ts, y*ts, ts, ts), 1)

    def draw_ui(self, surf, x, y):
        f = pygame.font.SysFont("consolas", 16)
        surf.blit(f.render(f"Motivation: {self.motivation.name}", True, (255, 255, 255)), (x, y))
        bar_w, bar_h = 100, 10
        # дома
        pygame.draw.rect(surf, (80, 80, 80), (x, y+20, bar_w, bar_h))
        pygame.draw.rect(surf, (0, 200, 0), (x, y+20, min(self.steps_in_home, bar_w), bar_h))
        # улица
        pygame.draw.rect(surf, (80, 80, 80), (x, y+35, bar_w, bar_h))
        pygame.draw.rect(surf, (0, 200, 200), (x, y+35, min(self.steps_outside, bar_w), bar_h))

    # -- private --
    def _count_steps(self):
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
        elif not self.path:
            self.motivation = Motivation.EXPLORE

    def _choose_goal(self) -> Optional[Pos]:
        if self.motivation == Motivation.EXPLORE:
            fr = [c for c in self.mem.frontier() if c != self.pos]
            return min(fr, key=lambda c: _h(self.pos, c)) if fr else None
        if self.motivation == Motivation.GO_HOME and self.zones.get("home"):
            return min(self.zones["home"], key=lambda c: _h(self.pos, c))
        if self.motivation == Motivation.GO_OUTSIDE:
            fr_out = [c for c in self.mem.frontier() if c not in self.zones.get("home", [])]
            return min(fr_out, key=lambda c: _h(self.pos, c)) if fr_out else None
        return None
