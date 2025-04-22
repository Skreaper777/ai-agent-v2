"""game-walls-tiled-1.py — запуск игры с Амелией‑ботом и Адамом
Полная версия, совместимая с актуальным agent_motivated.py
"""

import os; os.environ["SDL_VIDEO_WINDOW_POS"] = "1800,100"
import sys, math, json
from pathlib import Path
import pygame

from agent_motivated import Agent

# -----------------------------------------------------------------------------
# 1️⃣ Конфиг и константы
# -----------------------------------------------------------------------------
CFG = json.loads(Path("config.json").read_text(encoding="utf-8"))
WIN_W, WIN_H, SIDE_W = CFG["window_size"]["width"], CFG["window_size"]["height"], CFG["side_panel_width"]
BG_COLOR             = tuple(CFG["background_color"])

TILE_W, TILE_H       = CFG["tile_size"]["w"], CFG["tile_size"]["h"]
TILESET_PATH         = CFG["tileset_path"]

# спрайты
ADAM_SPRITE   = CFG["adam_sprite_path"]
AGENT_SPRITE  = CFG["agent_sprite_path"]

# стартовые позиции
ADAM_COL, ADAM_ROW   = CFG["adam_start"]["col"],  CFG["adam_start"]["row"]
AGENT_COL, AGENT_ROW = CFG["agent_start"]["col"], CFG["agent_start"]["row"]

# скорости (пикселей в кадр при 60 FPS)
ADAM_SPEED  = CFG["player_speed"]
AGENT_SPEED = CFG["agent_speed"]

# -----------------------------------------------------------------------------
# 2️⃣ Инициализация Pygame
# -----------------------------------------------------------------------------
pygame.init()
screen = pygame.display.set_mode((WIN_W+SIDE_W, WIN_H))
pygame.display.set_caption("Комната, Адам и Амелия‑бот")
font  = pygame.font.SysFont("consolas", 16)
clock = pygame.time.Clock()

# -----------------------------------------------------------------------------
# 3️⃣ Загрузка карты Tiled
# -----------------------------------------------------------------------------
MAP = json.loads(Path("map.json").read_text(encoding="utf-8"))
COLS, ROWS = MAP["width"], MAP["height"]
layer_tiles = next(l for l in MAP["layers"] if l["type"]=="tilelayer")
map_gids    = layer_tiles["data"]

ts_meta   = MAP["tilesets"][0]
first_gid = ts_meta["firstgid"]; cols_ts = ts_meta["columns"]
tileset_img = pygame.image.load(TILESET_PATH).convert_alpha()

# тайловые Surface'ы
tiles={}
for gid in {g for g in map_gids if g}:
    idx = gid-first_gid
    sx,sy = (idx%cols_ts)*TILE_W, (idx//cols_ts)*TILE_H
    tiles[gid] = tileset_img.subsurface((sx,sy,TILE_W,TILE_H))

def blocked(col,row): return map_gids[row*COLS+col]!=0

# зоны
zone_rects={}
for l in MAP["layers"]:
    if l["type"]=="objectgroup" and l["name"]=="zones":
        for obj in l["objects"]:
            for prop in obj.get("properties", []):
                if prop["name"]=="zone":
                    zone_rects[prop["value"]]=pygame.Rect(obj["x"],obj["y"],obj["width"],obj["height"])

# -----------------------------------------------------------------------------
# 4️⃣ Спрайты Адама и Амелии
# -----------------------------------------------------------------------------
FRAME_W, FRAME_H = 16, 32              # исходный размер персонажей
ASCALE = TILE_W // 16                 # масштаб до ширины тайла (48//16=3)
SW, SH  = FRAME_W*ASCALE, FRAME_H*ASCALE

# игрок (Адам)
adam_sheet = pygame.image.load(ADAM_SPRITE).convert_alpha()
adam_frames=[[pygame.transform.scale(adam_sheet.subsurface(((d*6+i)*FRAME_W,0,FRAME_W,FRAME_H)),(SW,SH))
              for i in range(6)] for d in range(4)]

adam_x = ADAM_COL*TILE_W+TILE_W//2
adam_y = ADAM_ROW*TILE_H+TILE_H//2
adam_dir, adam_frame, anim_t = 3,0,0.0

# агент (Амелия)
agent_sheet = pygame.image.load(AGENT_SPRITE).convert_alpha()
agent_frames=[[pygame.transform.scale(agent_sheet.subsurface(((d*6+i)*16,0,16,16)),(SW,SH))
               for i in range(6)] for d in range(4)]

# -----------------------------------------------------------------------------
# 5️⃣ Адаптер мира + правила + создание агента
# -----------------------------------------------------------------------------
class World:
    def __init__(self):
        self.size = (COLS, ROWS)
        self.ts   = TILE_W
    def in_bounds(self,p):  return 0<=p[0]<COLS and 0<=p[1]<ROWS
    def is_blocked(self,p): return blocked(*p)
world = World()

rules = json.loads(Path("agent_rules.json").read_text(encoding="utf-8"))

home_tiles=[]
if (rect:=zone_rects.get("home")):
    for r in range(ROWS):
        for c in range(COLS):
            if rect.collidepoint(c*TILE_W+TILE_W//2,r*TILE_H+TILE_H//2):
                home_tiles.append((c,r))

agent = Agent((AGENT_COL,AGENT_ROW), world, {"home":home_tiles},
              rules, agent_frames, AGENT_SPEED)
agent.load_memory()

# -----------------------------------------------------------------------------
# 6️⃣ Вспомогательные функции
# -----------------------------------------------------------------------------
def feet_ok(px,py):
    rect=pygame.Rect(px-SW*0.25, py-SH*0.25, SW*0.5, SH*0.25)
    c0,c1=max(0,rect.left//TILE_W),min(COLS,rect.right//TILE_W+1)
    r0,r1=max(0,rect.top //TILE_H),min(ROWS,rect.bottom//TILE_H+1)
    for r in range(r0,r1):
        for c in range(c0,c1):
            if blocked(c,r): return False
    return True

# -----------------------------------------------------------------------------
# 7️⃣ Игровой цикл
# -----------------------------------------------------------------------------
running=True
while running:
    dt=clock.tick(60)/1000
    for ev in pygame.event.get():
        if ev.type==pygame.QUIT: running=False
        if ev.type==pygame.KEYDOWN and ev.key==pygame.K_m:
            agent.save_memory()

    # --- управление Адамом ---
    k=pygame.key.get_pressed()
    vx=(k[pygame.K_RIGHT]-k[pygame.K_LEFT])*ADAM_SPEED
    vy=(k[pygame.K_DOWN ]-k[pygame.K_UP  ])*ADAM_SPEED
    if vx and vy:
        n=math.hypot(vx,vy); vx,vy=vx/n*ADAM_SPEED,vy/n*ADAM_SPEED
    if vx>0: adam_dir=0
    elif vx<0: adam_dir=2
    elif vy<0: adam_dir=1
    elif vy>0: adam_dir=3
    if feet_ok(adam_x+vx, adam_y+vy): adam_x+=vx; adam_y+=vy

    if vx or vy:
        anim_t+=dt*8
        if anim_t>1: anim_t=0; adam_frame=(adam_frame+1)%6
    else: adam_frame=0

    # --- логика агента ---
    agent.tick(dt)

    # -------------------------------------------------------------------------
    # 8️⃣  Рендер
    # -------------------------------------------------------------------------
    screen.fill(BG_COLOR)
    for r in range(ROWS):
        for c in range(COLS):
            gid=map_gids[r*COLS+c]
            if gid: screen.blit(tiles[gid],(c*TILE_W,r*TILE_H))

    if rect:=zone_rects.get("home"):
        s=pygame.Surface(rect.size,pygame.SRCALPHA); s.fill((0,0,255,60))
        screen.blit(s,rect.topleft)
        screen.blit(font.render("дом",True,(255,255,255)),(rect.x+4,rect.y+4))

    # персонажи
    screen.blit(adam_frames[adam_dir][adam_frame],(adam_x-SW//2,adam_y-SH))
    agent.draw(screen); agent.draw_path(screen,TILE_W)

    # UI
    screen.blit(font.render(f"Мотивация: {agent.motivation.value}",True,(255,255,255)),(WIN_W+10,20))

    # мини‑карта памяти
    g=agent.mem.grid; mini=pygame.Surface((g.shape[1],g.shape[0]))
    for y in range(g.shape[0]):
        for x in range(g.shape[1]):
            v=g[y,x]; mini.set_at((x,y),(40,40,40) if v==-1 else (100,100,100) if v==0 else (180,180,255))
    screen.blit(pygame.transform.scale(mini,(g.shape[1]*2,g.shape[0]*2)),(WIN_W+10,60))

    pygame.display.flip()

pygame.quit(); sys.exit()
