"""game-walls-tiled-1.py — главный исполняемый файл
===================================================
Все комментарии переведены на русский язык.
Игра: Адам (игрок) + умный агент‑исследователь.
"""

import os
os.environ['SDL_VIDEO_WINDOW_POS'] = "1800,100"  # позиция окна (можно убрать)

# --- стандартные и сторонние библиотеки ---
import sys
import math
import json
from pathlib import Path

import pygame

# --- наши модули ---
from agent_motivated import Agent  # умный агент

# -----------------------------------------------------------------------------
# 1. Загрузка конфигурации
# -----------------------------------------------------------------------------
CFG = json.loads(Path("config.json").read_text(encoding="utf-8"))
WIN_W, WIN_H = CFG["window_size"]["width"], CFG["window_size"]["height"]
SIDE_W       = CFG.get("side_panel_width", 200)
BG_COLOR     = tuple(CFG["background_color"])
TILE_W, TILE_H = CFG["tile_size"]["w"], CFG["tile_size"]["h"]
TILESET_PATH   = CFG["tileset_path"]
ADAM_SPRITE    = CFG["adam_sprite_path"]
ADAM_COL, ADAM_ROW   = CFG["adam_start"]["col"], CFG["adam_start"]["row"]
AGENT_COL, AGENT_ROW = CFG["agent_start"]["col"], CFG["agent_start"]["row"]

# -----------------------------------------------------------------------------
# 2. Инициализация Pygame (должна быть ДО загрузки Surface'ов)
# -----------------------------------------------------------------------------
pygame.init()
screen = pygame.display.set_mode((WIN_W + SIDE_W, WIN_H))
pygame.display.set_caption("Комната, Адам и умный агент")
font  = pygame.font.SysFont("consolas", 16)
clock = pygame.time.Clock()

# -----------------------------------------------------------------------------
# 3. Загрузка карты Tiled (.json)
# -----------------------------------------------------------------------------
MAP = json.loads(Path("map.json").read_text(encoding="utf-8"))
MAP_COLS, MAP_ROWS = MAP["width"], MAP["height"]
assert (MAP["tilewidth"], MAP["tileheight"]) == (TILE_W, TILE_H), "Размер тайлов не совпадает с конфигом"

layer_tiles = next(l for l in MAP["layers"] if l["type"] == "tilelayer")
map_gids    = layer_tiles["data"]

ts_meta   = MAP["tilesets"][0]
first_gid = ts_meta["firstgid"]
cols_ts   = ts_meta["columns"]
img_tiles = pygame.image.load(TILESET_PATH).convert_alpha()

tile_surfs: dict[int, pygame.Surface] = {}
for gid in {g for g in map_gids if g}:
    idx = gid - first_gid
    src_x = (idx % cols_ts) * TILE_W
    src_y = (idx // cols_ts) * TILE_H
    tile_surfs[gid] = img_tiles.subsurface((src_x, src_y, TILE_W, TILE_H))

# --- коллизии (objectgroup внутри tileset) ---
coll_shapes: dict[int, list[pygame.Rect]] = {}
for tile in ts_meta.get("tiles", []):
    gid  = first_gid + tile["id"]
    objs = tile.get("objectgroup", {}).get("objects", [])
    if objs:
        coll_shapes[gid] = [pygame.Rect(o["x"], o["y"], o["width"], o["height"]) for o in objs]

# --- зоны (слой zones) ---
zone_rects: dict[str, pygame.Rect] = {}
for layer in MAP["layers"]:
    if layer["type"] == "objectgroup" and layer["name"] == "zones":
        for obj in layer["objects"]:
            for prop in obj.get("properties", []):
                if prop["name"] == "zone":
                    zone_rects[prop["value"]] = pygame.Rect(obj["x"], obj["y"], obj["width"], obj["height"])

# -----------------------------------------------------------------------------
# 4. Игрок: спрайты и анимация
# -----------------------------------------------------------------------------
FRAME_W, FRAME_H = 16, 32
SCALE = 2
SW, SH = FRAME_W * SCALE, FRAME_H * SCALE
ANIM_SPEED   = 0.12
PLAYER_SPEED = 2

adam_sheet = pygame.image.load(ADAM_SPRITE).convert_alpha()

def slice_adam(sheet: pygame.Surface) -> list[list[pygame.Surface]]:
    frames = [[] for _ in range(4)]  # 0‑right 1‑up 2‑left 3‑down
    for d in range(4):
        for i in range(6):
            surf = sheet.subsurface(((d*6+i)*FRAME_W, 0, FRAME_W, FRAME_H))
            frames[d].append(pygame.transform.scale(surf, (SW, SH)))
    return frames

adam_frames = slice_adam(adam_sheet)

adam_x = ADAM_COL * TILE_W + TILE_W//2
adam_y = ADAM_ROW * TILE_H + TILE_H//2

# -----------------------------------------------------------------------------
# 5. Адаптер мира и создание агента
# -----------------------------------------------------------------------------
class WorldAdapter:
    """Интерфейс между картой и Agent."""
    def __init__(self, gids, coll, w, h):
        self.gids, self.coll, self.size = gids, coll, (w, h)
    def in_bounds(self, pos):
        x, y = pos
        return 0 <= x < self.size[0] and 0 <= y < self.size[1]
    def is_blocked(self, pos):
        x, y = pos
        gid = self.gids[y*self.size[0] + x]
        return gid in self.coll

world = WorldAdapter(map_gids, coll_shapes, MAP_COLS, MAP_ROWS)

home_tiles: list[tuple[int,int]] = []
if (home_rect := zone_rects.get("home")):
    for ty in range(MAP_ROWS):
        for tx in range(MAP_COLS):
            if home_rect.collidepoint(tx*TILE_W + TILE_W//2, ty*TILE_H + TILE_H//2):
                home_tiles.append((tx, ty))

agent = Agent((AGENT_COL, AGENT_ROW), world, {"home": home_tiles})

# -----------------------------------------------------------------------------
# 6. Переменные анимации
# -----------------------------------------------------------------------------
frame_idx = 0
anim_t    = 0.0
adam_dir  = 3  # 0‑right 1‑up 2‑left 3‑down

# -----------------------------------------------------------------------------
# 7. Основной игровой цикл
# -----------------------------------------------------------------------------
running = True
while running:

    # ==== ограничиваем FPS, получаем dt ====
    dt = clock.tick(60) / 1000.0

    # ==== обработка событий ====
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False

    # ==== ввод игрока ====
    keys = pygame.key.get_pressed()
    vx = vy = 0
    if keys[pygame.K_LEFT]:  vx -= PLAYER_SPEED
    if keys[pygame.K_RIGHT]: vx += PLAYER_SPEED
    if keys[pygame.K_UP]:    vy -= PLAYER_SPEED
    if keys[pygame.K_DOWN]:  vy += PLAYER_SPEED
    if vx and vy:  # нормализация диагонали
        n = math.hypot(vx, vy)
        vx, vy = vx/n*PLAYER_SPEED, vy/n*PLAYER_SPEED

    if vx>0: adam_dir=0
    elif vx<0: adam_dir=2
    elif vy<0: adam_dir=1
    elif vy>0: adam_dir=3

    # ==== простая коллизия по «ногам» персонажа ====
    def foot_collides(rect: pygame.Rect) -> bool:
        c0 = max(0, rect.left // TILE_W)
        c1 = min(MAP_COLS, rect.right // TILE_W + 1)
        r0 = max(0, rect.top // TILE_H)
        r1 = min(MAP_ROWS, rect.bottom // TILE_H + 1)
        for r in range(r0, r1):
            for c in range(c0, c1):
                gid = map_gids[r*MAP_COLS + c]
                if gid == 0:
                    continue
                if gid in coll_shapes:
                    for s in coll_shapes[gid]:
                        if rect.colliderect(pygame.Rect(c*TILE_W+s.x, r*TILE_H+s.y, s.w, s.h)):
                            return True
                else:
                    if rect.colliderect(pygame.Rect(c*TILE_W, r*TILE_H, TILE_W, TILE_H)):
                        return True
        return False

    foot_w, foot_h = SW*0.5, SH*0.25
    next_rect = pygame.Rect(adam_x + vx - foot_w/2, adam_y + vy - foot_h, foot_w, foot_h)
    if not foot_collides(next_rect):
        adam_x += vx
        adam_y += vy

    # ==== обновляем агента ====
    agent.tick()

    # ==== анимация игрока ====
    moving = vx or vy
    if moving:
        anim_t += ANIM_SPEED
        if anim_t >= 1:
            anim_t = 0
            frame_idx = (frame_idx + 1) % 6
    else:
        frame_idx = 0

    # -------------------------------------------------------------------------
    # Рендер
    # -------------------------------------------------------------------------
    screen.fill(BG_COLOR)

    # --- тайловый слой ---
    for r in range(MAP_ROWS):
        for c in range(MAP_COLS):
            if (img := tile_surfs.get(map_gids[r*MAP_COLS + c])):
                screen.blit(img, (c*TILE_W, r*TILE_H))

    # --- зона «home» полупрозрачным прямоугольником ---
    if home_rect:
        h_surf = pygame.Surface((home_rect.w, home_rect.h), pygame.SRCALPHA)
        h_surf.fill((0, 0, 255, 60))
        screen.blit(h_surf, home_rect.topleft)
        screen.blit(font.render("home", True, (255,255,255)), (home_rect.x+4, home_rect.y+4))

    # --- спрайт игрок

