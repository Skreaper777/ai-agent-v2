"""agent_motivated.py — ИИ‑агент со спрайтом и гибкими мотивациями"""
import json, heapq, random
from enum   import Enum, auto
from typing import Dict, List, Optional, Tuple

import pygame, numpy as np

Pos = Tuple[int, int]                      # (col,row) в тайлах

# ---------- мотивация ----------
class Motivation(Enum):
    EXPLORE   = "ИССЛЕДОВАТЬ"
    GO_HOME   = "ИДТИ_ДОМОЙ"
    GO_STROLL = "ИДТИ_НАРУЖУ"

# ---------- память ----------
class MapMemory:
    def __init__(self, size: Tuple[int, int]):
        w, h      = size
        self.grid = np.full((h, w), -1, dtype=np.int8)   # -1 ? , 0 стена, 1 пол

    def update(self, world, pos: Pos, r: int = 2):
        cx, cy = pos
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                nx, ny = cx + dx, cy + dy
                if world.in_bounds((nx, ny)):
                    self.grid[ny, nx] = 0 if world.is_blocked((nx, ny)) else 1

    def __getitem__(self, p: Pos) -> int:
        x, y = p
        h, w = self.grid.shape
        if 0 <= x < w and 0 <= y < h:
            return self.grid[y, x]
        return 0                                    # за пределами — стена

    # клетки‑фронтиры
    def frontier(self) -> List[Pos]:
        out: List[Pos] = []
        h, w = self.grid.shape
        for y in range(h):
            for x in range(w):
                if self.grid[y, x] != 1: continue
                for nx, ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                    if 0 <= nx < w and 0 <= ny < h and self.grid[ny, nx] == -1:
                        out.append((x, y)); break
        return out

    # сериализация для сохранения
    def to_json(self) -> list:  return self.grid.tolist()
    def load_json(self, data):   self.grid[:] = np.array(data, dtype=np.int8)

# ---------- A* ----------
def _h(a: Pos, b: Pos) -> int:  return abs(a[0]-b[0]) + abs(a[1]-b[1])
def _astar(mem: MapMemory, s: Pos, g: Pos) -> List[Pos]:
    open_ = [(0+_h(s,g),0,s,None)]
    came: Dict[Pos, Optional[Pos]] = {}; g_score = {s:0}
    while open_:
        _, g_cost, cur, parent = heapq.heappop(open_)
        if cur in came:         continue
        came[cur] = parent
        if cur == g:
            path = [cur]
            while came[path[-1]] is not None: path.append(came[path[-1]])
            return path[::-1]
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx,ny = cur[0]+dx, cur[1]+dy
            if mem[(nx,ny)] != 1:   continue
            ng = g_cost+1
            if ng < g_score.get((nx,ny),9e9):
                g_score[(nx,ny)] = ng
                heapq.heappush(open_, (ng+_h((nx,ny),g), ng, (nx,ny), cur))
    return []

# ---------- сам агент ----------
class Agent:
    def __init__(self,
                 start_tile : Pos,
                 world,
                 zones      : Dict[str, List[Pos]],
                 rules      : Dict[str, float],
                 sprite_set : List[List[pygame.Surface]],
                 speed_px   : float):
        # положение храним и в тайлах, и в пикселях
        self.tile = list(start_tile)                        # [col,row]
        self.px   = [start_tile[0]*world.ts+world.ts//2,
                     start_tile[1]*world.ts+world.ts//2]
        self.speed_px = speed_px
        self.dir_idx  = 3
        self.anim_t   = 0
        self.frame    = 0
        self.frames   = sprite_set

        self.world = world
        self.zones = zones
        self.mem   = MapMemory(world.size)
        self.path: List[Pos] = []

        self.rules_out = int(rules["outside_timeout"] * 60)  # перевод сек→кадры
        self.rules_home= int(rules["home_timeout"]    * 60)

        self.motivation = Motivation.EXPLORE
        self.step_out, self.step_in = 0, 0

    # ---------- логика ----------
    def tick(self, dt):
        # обновляем память
        self.mem.update(self.world, tuple(self.tile))

        # счётчики
        if tuple(self.tile) in self.zones.get("home", []):
            self.step_in += 1; self.step_out = 0
        else:
            self.step_out += 1; self.step_in = 0

        # смена мотивации
        if self.step_in  > self.rules_home:  self.motivation = Motivation.GO_STROLL
        if self.step_out > self.rules_out:   self.motivation = Motivation.GO_HOME
        if not self.path:                    self.motivation = Motivation.EXPLORE

        # если нет пути — строим
        if not self.path:
            goal = self._choose_goal()
            if goal:  self.path = _astar(self.mem, tuple(self.tile), goal)[1:]

        # ---------- движение по пикселям ----------
        if self.path:
            nxt = self.path[0]
            tgt_px = (nxt[0]*self.world.ts+self.world.ts//2,
                      nxt[1]*self.world.ts+self.world.ts//2)
            vx = tgt_px[0]-self.px[0]; vy = tgt_px[1]-self.px[1]
            dist = (vx*vx+vy*vy)**0.5
            if dist < self.speed_px:               # достигли клетки
                self.px[:] = tgt_px
                self.tile[:] = nxt
                self.path.pop(0)
            else:                                  # шаг к цели
                self.px[0] += vx/dist*self.speed_px
                self.px[1] += vy/dist*self.speed_px
            # направление для анимации
            if abs(vx) > abs(vy):  self.dir_idx = 0 if vx>0 else 2
            else:                  self.dir_idx = 3 if vy>0 else 1

        # ---------- анимация ----------
        self.anim_t += dt*8
        if self.anim_t >= 1:
            self.anim_t = 0; self.frame = (self.frame+1)%6

    # ---------- выбор цели ----------
    def _choose_goal(self) -> Optional[Pos]:
        if self.motivation == Motivation.EXPLORE:
            frontier = [c for c in self.mem.frontier() if c != tuple(self.tile)]
            return min(frontier, key=lambda p:_h(p, self.tile)) if frontier else None
        if self.motivation == Motivation.GO_HOME and self.zones.get("home"):
            return min(self.zones["home"], key=lambda p:_h(p, self.tile))
        if self.motivation == Motivation.GO_STROLL:
            fr = [p for p in self.mem.frontier() if p not in self.zones.get("home", [])]
            return min(fr, key=lambda p:_h(p, self.tile)) if fr else None
        return None

    # ---------- рендер ----------
    def draw(self, surf):
        img = self.frames[self.dir_idx][self.frame]
        surf.blit(img, (self.px[0]-img.get_width()//2, self.px[1]-img.get_height()))

    def draw_path(self, surf, ts):
        for c,r in self.path:
            pygame.draw.rect(surf, (255,0,0), (c*ts, r*ts, ts, ts),1)

    # ---------- сохранение/загрузка памяти ----------
    def save_memory(self, path="agent_memory.json"):
        json.dump(self.mem.to_json(), open(path,"w"), indent=0)
    def load_memory(self, path="agent_memory.json"):
        try: self.mem.load_json(json.load(open(path)))
        except FileNotFoundError: pass
