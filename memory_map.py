# memory_map.py — компонент памяти агента для запоминания карты

import numpy as np

class MapMemory:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # -1 = неизвестно, 0 = непроходимо, 1 = проходимо
        self.memory = np.full((height, width), -1, dtype=int)

    def update_from_vision(self, cx, cy, vision):
        """
        cx, cy — текущая клетка агента
        vision — 5x5 матрица видимости (0 или 1)
        """
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                tx, ty = cx + dx, cy + dy
                if 0 <= tx < self.width and 0 <= ty < self.height:
                    self.memory[ty, tx] = vision[dy + 2][dx + 2]

    def is_known(self, x, y):
        return self.memory[y, x] != -1

    def get_value(self, x, y):
        return self.memory[y, x]

    def visualize(self):
        """Для отладки — вывод в консоль"""
        for row in self.memory:
            print("".join({-1: "?", 0: "#", 1: "."}[val] for val in row))

    def export_as_surface(self, tile_size):
        """
        Возвращает Pygame-сурфейс с мини-картой памяти
        tile_size — размер квадратика
        """
        import pygame
        surf = pygame.Surface((self.width * tile_size, self.height * tile_size))
        for y in range(self.height):
            for x in range(self.width):
                v = self.memory[y, x]
                color = (40, 40, 40) if v == -1 else (100, 100, 100) if v == 0 else (180, 180, 255)
                pygame.draw.rect(surf, color, (x*tile_size, y*tile_size, tile_size, tile_size))
        return surf