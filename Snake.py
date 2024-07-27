import pygame
import random

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 640
GRID_SIZE = 16
DIRECTION = [(1, 0), (-1, 0), (0, 1), (0, -1)]


class Snake:
    def __init__(self):
        self.reset()

    def move(self):
        head = self.body[0].copy()
        head.x += self.direction.x * GRID_SIZE
        head.y += self.direction.y * GRID_SIZE
        self.body.insert(0, head)
        
        if len(self.body) > self.length:
            self.body.pop()

    def is_aligned_with_grid(self):
        return self.body[0].x % GRID_SIZE == 0 and self.body[0].y % GRID_SIZE == 0

    def draw(self, screen):
        for segment in self.body:
            pygame.draw.rect(screen, (0, 255, 0), segment)

    def change_direction(self, new_direction):
        if (self.direction.x == -new_direction.x and self.direction.y == -new_direction.y):
            return

        if self.is_aligned_with_grid():
            self.direction = new_direction
            
    def check_borders(self):
        head = self.body[0]
        if head.x < 0:
            head.x = SCREEN_WIDTH - GRID_SIZE
            
        elif head.x >= SCREEN_WIDTH:
            head.x = 0
            
        elif head.y >= SCREEN_HEIGHT:
            head.y = 0

        elif head.y < 0:
            head.y = SCREEN_HEIGHT - GRID_SIZE

    def collision_detection(self):
        head = self.body[0]
        
        for segment in self.body[1:]:
            if head.colliderect(segment):
                return True
                
        return False

    def reset(self):
        self.body = [pygame.Rect(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, GRID_SIZE, GRID_SIZE)]
        self.direction = pygame.Vector2(DIRECTION[random.randint(0, 3)])
        self.length = 5
        self.collision = False

