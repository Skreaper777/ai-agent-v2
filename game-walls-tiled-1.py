"""game-walls-tiled-1.py — запуск игры с умным агентом"""
import os
os.environ["SDL_VIDEO_WINDOW_POS"] = "1800,100"

# --- библиотеки ---
import sys, math, json, random
from pathlib import Path
import pygame

from agent_motivated import Agent

# ------------------- Конфиг -------------------
CFG = json.loads(Path("config.json").read_text(encoding="utf-8"))
WIN_W, WIN_H = CFG["window_size"]["width"], CFG["window_size"]["height"]
SIDE_W       = CFG.get("side_panel_width", 200)
BG_COLOR     = tuple(CFG["background_color"])

TILE_W, TILE_H = CFG["tile_size"]["w"], CFG["tile_size"]["h"]
TILESET_PATH   = CFG["tileset_path"]
ADAM_SPRITE    = CFG["adam_sprite_path"]
ADAM_COL, ADAM_ROW   = CFG["adam_start"]["col"], CFG["adam_start"]["row"]
AGENT_COL, AGENT_ROW = CFG["agent_start"]["col"], CFG["agent_start"]["row"]

# ------------- Pygame -------------
pygame.init()
screen = pygame.display.set_mode((WIN_W + SIDE_W, WIN_H))
pygame.display.set_caption("Комната, Адам и умный агент")
font  = pygame.font.SysFont("consolas", 16)
clock = pygame.time.Clock()

# ------------------- Карта Tiled -------------------
MAP = json.loads(Path("map.json").read_text(encoding="utf-8"))
MAP_COLS, MAP_ROWS = MAP["width"], MAP["height"]

layer_tiles = next(l for l in MAP["layers"] if l["type"] == "tilelayer")
map_gids = layer_tiles["data"]

ts_meta   = MAP["tilesets"][0]
first_gid, cols_ts = ts_meta["firstgid"], ts_meta["columns"]
tileset_img = pygame.image.load(TILESET_PATH).convert_alpha()

tile_surfs: dict[int, pygame.Surface] = {}
for gid in {g for g in map_gids if g}:
    idx = gid - first_gid
    sx, sy = (idx % cols_ts)*TILE_W, (idx // cols_ts)*TILE_H
    tile_surfs[gid] = tileset_img.subsurface((sx, sy, TILE_W, TILE_H))

# --- коллизии ---
coll_shapes: dict[int, list[pygame.Rect]] = {}
for t in ts_meta.get("tiles", []):
    gid = first_gid + t["id"]
    objs = t.get("objectgroup", {}).get("objects", [])
    if objs:
        coll_shapes[gid] = [pygame.Rect(o["x"], o["y"], o["width"], o["height"]) for o in objs]

# --- зоны ---
zone_rects: dict[str, pygame.Rect] = {}
for l in MAP["layers"]:
    if l["type"] == "objectgroup" and l["name"] == "zones":
        for obj in l["objects"]:
            for prop in obj.get("properties", []):
                if prop["name"] == "zone":
                    zone_rects[prop["value"]] = pygame.Rect(obj["x"], obj["y"],
                                                            obj["width"], obj["height"])

# ------------------- Игрок -------------------
FRAME_W, FRAME_H, SCALE = 16, 32, 2
SW, SH = FRAME_W*SCALE, FRAME_H*SCALE
PLAYER_SPEED, ANIM_SPEED = 2, 0.12

adam_sheet = pygame.image.load(ADAM_SPRITE).convert_alpha()
adam_frames = [[pygame.transform.scale(
                    adam_sheet.subsurface(((d*6+i)*FRAME_W, 0, FRAME_W, FRAME_H)),
                    (SW, SH))
                for i in range(6)] for d in range(4)]

adam_x = ADAM_COL*TILE_W + TILE_W//2
adam_y = ADAM_ROW*TILE_H + TILE_H//2
frame_idx, anim_t, adam_dir = 0, 0.0, 3

# ------------------- Мир + агент -------------------
class WorldAdapter:
    def __init__(self, gids, coll, w, h):
        self.gids, self.coll, self.size = gids, coll, (w, h)
    def in_bounds(self, p):  return 0 <= p[0] < self.size[0] and 0 <= p[1] < self.size[1]
    def is_blocked(self, p): return self.gids[p[1]*self.size[0] + p[0]] in self.coll

world = WorldAdapter(map_gids, coll_shapes, MAP_COLS, MAP_ROWS)

home_tiles = []
if (home_rect := zone_rects.get("home")):
    for ty in range(MAP_ROWS):
        for tx in range(MAP_COLS):
            if home_rect.collidepoint(tx*TILE_W+TILE_W//2, ty*TILE_H+TILE_H//2):
                home_tiles.append((tx, ty))

agent = Agent((AGENT_COL, AGENT_ROW), world, {"home": home_tiles})

# ------------------- Игровой цикл -------------------
running = True
while running:
    dt = clock.tick(60)/1000.0

    # -- события --
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False

    # -- управление игроком --
    keys = pygame.key.get_pressed()
    vx = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * PLAYER_SPEED
    vy = (keys[pygame.K_DOWN]  - keys[pygame.K_UP])   * PLAYER_SPEED
    if vx and vy:
        n = math.hypot(vx, vy); vx, vy = vx/n*PLAYER_SPEED, vy/n*PLAYER_SPEED

    if vx>0: adam_dir=0
    elif vx<0: adam_dir=2
    elif vy<0: adam_dir=1
    elif vy>0: adam_dir=3

    # -- коллизия ног --
    def feet_ok(nx, ny):
        rect = pygame.Rect(nx - SW*0.25, ny - SH*0.25, SW*0.5, SH*0.25)
        c0, c1 = max(0, rect.left//TILE_W), min(MAP_COLS, rect.right//TILE_W+1)
        r0, r1 = max(0, rect.top //TILE_H), min(MAP_ROWS, rect.bottom//TILE_H+1)
        for r in range(r0, r1):
            for c in range(c0, c1):
                gid = map_gids[r*MAP_COLS+c]
                if gid==0: continue
                if gid in coll_shapes:
                    for s in coll_shapes[gid]:
                        if rect.colliderect(pygame.Rect(c*TILE_W+s.x, r*TILE_H+s.y, s.w, s.h)):
                            return False
                else:
                    if rect.colliderect(pygame.Rect(c*TILE_W, r*TILE_H, TILE_W, TILE_H)):
                        return False
        return True

    if feet_ok(adam_x+vx, adam_y+vy):
        adam_x += vx; adam_y += vy

    # -- логика агента --
    agent.tick()

    # -- анимация игрока --
    moving = vx or vy
    if moving:
        anim_t += ANIM_SPEED
        if anim_t >= 1: anim_t, frame_idx = 0, (frame_idx+1)%6
    else:
        frame_idx = 0

    # ------------------- Рендер -------------------
    screen.fill(BG_COLOR)

    # тайлы
    for r in range(MAP_ROWS):
        for c in range(MAP_COLS):
            gid = map_gids[r*MAP_COLS + c]
            if gid and gid in tile_surfs:
                screen.blit(tile_surfs[gid], (c*TILE_W, r*TILE_H))

    # зона home
    if home_rect:
        tmp = pygame.Surface(home_rect.size, pygame.SRCALPHA)
        tmp.fill((0,0,255,60))
        screen.blit(tmp, home_rect.topleft)
        screen.blit(font.render("home", True, (255,255,255)), (home_rect.x+4, home_rect.y+4))

    # спрайт игрока
    screen.blit(adam_frames[adam_dir][frame_idx], (adam_x-SW//2, adam_y-SH))

    # путь и позиция агента
    agent.draw_path(screen, TILE_W)
    pygame.draw.rect(screen, (255,0,0),
                     (agent.pos[0]*TILE_W, agent.pos[1]*TILE_H, TILE_W, TILE_H), 2)

    # UI
    agent.draw_ui(screen, WIN_W+10, 20)

    # мини‑карта памяти
    mem = agent.mem.grid
    mini = pygame.Surface((mem.shape[1], mem.shape[0]))
    for y in range(mem.shape[0]):
        for x in range(mem.shape[1]):
            v = mem[y,x]
            mini.set_at((x,y), (40,40,40) if v==-1 else (100,100,100) if v==0 else (180,180,255))
    screen.blit(pygame.transform.scale(mini, (mem.shape[1]*2, mem.shape[0]*2)), (WIN_W+10, 120))

    pygame.display.flip()

pygame.quit()
sys.exit()
