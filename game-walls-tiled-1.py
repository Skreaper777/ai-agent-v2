import os
os.environ['SDL_VIDEO_WINDOW_POS'] = "1800,100"  # позиция окна (можно убрать)

# --- std & third‑party ---
import sys, math, json, pygame
from pathlib import Path

# --- project local ---
from agent_motivated import Agent

# -------------------------------------------------
# 1️⃣ Загрузка конфигурации
# -------------------------------------------------
CFG = Path("config.json").read_text(encoding="utf-8")
cfg = json.loads(CFG)

WIN_W, WIN_H    = cfg["window_size"]["width"], cfg["window_size"]["height"]
SIDE_W          = cfg.get("side_panel_width", 200)
BG_COLOR        = tuple(cfg["background_color"])
TILE_W, TILE_H  = cfg["tile_size"]["w"], cfg["tile_size"]["h"]
TILESET_PATH    = cfg["tileset_path"]
ADAM_SPRITE     = cfg["adam_sprite_path"]

# -------------------------------------------------
# 2️⃣ Инициализация pygame ДО загрузки картинок!
# -------------------------------------------------
pygame.init()
screen = pygame.display.set_mode((WIN_W + SIDE_W, WIN_H))
pygame.display.set_caption("Room & Adam with Smart Agent")
font = pygame.font.SysFont("consolas", 16)
clock = pygame.time.Clock()

# -------------------------------------------------
# 3️⃣ Загрузка карты из Tiled
# -------------------------------------------------
map_data = json.loads(Path("map.json").read_text(encoding="utf-8"))
MAP_COLS, MAP_ROWS = map_data["width"], map_data["height"]
assert (map_data["tilewidth"], map_data["tileheight"]) == (TILE_W, TILE_H), "Tile size mismatch"

layer_tiles = next(l for l in map_data["layers"] if l["type"] == "tilelayer")
map_gids    = layer_tiles["data"]

ts_meta   = map_data["tilesets"][0]
first_gid = ts_meta["firstgid"]
cols_ts   = ts_meta["columns"]
img_tileset = pygame.image.load(TILESET_PATH).convert_alpha()

tiles: dict[int, pygame.Surface] = {}
for gid in {g for g in map_gids if g}:  # уникальные ненулевые gid
    idx = gid - first_gid
    x0 = (idx % cols_ts) * TILE_W
    y0 = (idx // cols_ts) * TILE_H
    tiles[gid] = img_tileset.subsurface(pygame.Rect(x0, y0, TILE_W, TILE_H))

# --- коллизии ---
coll_shapes: dict[int, list[pygame.Rect]] = {}
for tile in ts_meta.get("tiles", []):
    gid = first_gid + tile["id"]
    objs = tile.get("objectgroup", {}).get("objects", [])
    if objs:
        coll_shapes[gid] = [pygame.Rect(o["x"], o["y"], o["width"], o["height"]) for o in objs]

# --- зоны ---
zone_rects: dict[str, pygame.Rect] = {}
for l in map_data["layers"]:
    if l["type"] == "objectgroup" and l["name"] == "zones":
        for obj in l["objects"]:
            for prop in obj.get("properties", []):
                if prop["name"] == "zone":
                    zone_rects[prop["value"]] = pygame.Rect(obj["x"], obj["y"], obj["width"], obj["height"])

# -------------------------------------------------
# 4️⃣ Игрок — спрайты и старт
# -------------------------------------------------
FRAME_W, FRAME_H = 16, 32
SCALE = 2
SW, SH = FRAME_W*SCALE, FRAME_H*SCALE
ANIM_SPEED = 0.12
PLAYER_SPEED = 2

adam_sheet = pygame.image.load(ADAM_SPRITE).convert_alpha()

def slice_adam(sheet: pygame.Surface):
    frames = [[] for _ in range(4)]
    for d in range(4):
        for i in range(6):
            surf = sheet.subsurface(pygame.Rect((d*6+i)*FRAME_W, 0, FRAME_W, FRAME_H))
            frames[d].append(pygame.transform.scale(surf, (SW, SH)))
    return frames

adam_frames = slice_adam(adam_sheet)

ADAM_COL, ADAM_ROW = cfg["adam_start"]["col"], cfg["adam_start"]["row"]
AGENT_COL, AGENT_ROW = cfg["agent_start"]["col"], cfg["agent_start"]["row"]

adam_x = ADAM_COL * TILE_W + TILE_W//2
adam_y = ADAM_ROW * TILE_H + TILE_H//2

# -------------------------------------------------
# 5️⃣ Адаптер мира + агент
# -------------------------------------------------
class WorldAdapter:
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

home_tiles = []
if (home_rect := zone_rects.get("home")):
    for ty in range(MAP_ROWS):
        for tx in range(MAP_COLS):
            if home_rect.collidepoint(tx*TILE_W + TILE_W//2, ty*TILE_H + TILE_H//2):
                home_tiles.append((tx, ty))

agent = Agent((AGENT_COL, AGENT_ROW), world, {"home": home_tiles})

# -------------------------------------------------
# 6️⃣ Основной цикл
# -------------------------------------------------
frame_idx = 0
anim_t = 0.0
dir_idx = 3  # 3‑down

running = True
while running:
    dt = clock.tick(60) / 1000.0
    # --- events
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False

    # --- input
    keys = pygame.key.get_pressed()
    vx = vy = 0
    if keys[pygame.K_LEFT]:   vx -= PLAYER_SPEED
    if keys[pygame.K_RIGHT]:  vx += PLAYER_SPEED
    if keys[pygame.K_UP]:     vy -= PLAYER_SPEED
    if keys[pygame.K_DOWN]:   vy += PLAYER_SPEED
    if vx and vy:
        n = math.hypot(vx, vy)
        vx, vy = vx/n*PLAYER_SPEED, vy/n*PLAYER_SPEED

    if vx>0: dir_idx=0
    elif vx<0: dir_idx=2
    elif vy<0: dir_idx=1
    elif vy>0: dir_idx=3

    # --- simple collision for player (feet box)
    def rect_collides(rect: pygame.Rect):
        c0 = max(0, rect.left // TILE_W)
        c1 = min(MAP_COLS, rect.right // TILE_W + 1)
        r0 = max(0, rect.top // TILE_H)
        r1 = min(MAP_ROWS, rect.bottom // TILE_H + 1)
        for r in range(r0, r1):
            for c in range(c0, c1):
                gid = map_gids[r*MAP_COLS + c]
                if gid == 0: continue
                shapes = coll_shapes.get(gid)
                if shapes:
                    for s in shapes:
                        if rect.colliderect(pygame.Rect(c*TILE_W + s.x, r*TILE_H + s.y, s.w, s.h)):
                            return True
                else:
                    if rect.colliderect(pygame.Rect(c*TILE_W, r*TILE_H, TILE_W, TILE_H)):
                        return True
        return False

    feet_w, feet_h = SW*0.5, SH*0.25
    next_rect = pygame.Rect(adam_x + vx - feet_w/2, adam_y + vy - feet_h, feet_w, feet_h)
    if not rect_collides(next_rect):
        adam_x += vx
        adam_y += vy

    # --- agent update
    agent.tick()

    # --- animation
    moving = vx or vy
    if moving:
        anim_t += ANIM_SPEED
        if anim_t >= 1:
            anim_t = 0
            frame_idx = (frame_idx + 1) % 6
    else:
        frame_idx = 0

    # -------------------------------------------------
    # 7️⃣ Рендер
    # -------------------------------------------------
    screen.fill(BG_COLOR)
    for r in range(MAP_ROWS):
        for c in range(MAP_COLS):
            if (img := tiles.get(map_gids[r*MAP_COLS + c])):
                screen.blit(img, (c*TILE_W, r*TILE_H))

    # zone overlay
    if home_rect:
        surf = pygame.Surface((home_rect.w, home_rect.h), pygame.SRCALPHA)
        surf.fill((0,0,255,60))
        screen.blit(surf, home_rect.topleft)
        screen.blit(font.render("home", True, (255,255,255)), (home_rect.x+4, home_rect.y+4))

    screen.blit(adam_frames[dir_idx][frame_idx], (adam_x - SW/2, adam_y - SH))

    agent.draw_path(screen, TILE_W)
    pygame.draw.rect(screen, (255,0,0), (agent.pos[0]*TILE_W, agent.pos[1]*TILE_H, TILE_W, TILE_H), 2)

    # UI
    panel_x = WIN_W + 10
    agent.draw_ui(screen, panel_x, 20)

    # mini‑memory map
    mem = agent.mem._grid
    h, w = mem.shape
    mini = pygame.Surface((w, h))
    for y in range(h):
        for x in range(w):
            v = mem[y,x]
            mini.set_at((x,y), (40,40,40) if v==-1 else (100,100,100) if v==0 else (180,180,255))
    screen.blit(pygame.transform.scale(mini, (w*2, h*2)), (panel_x, 120))

    pygame.display.flip()

pygame.quit()
sys.exit()
