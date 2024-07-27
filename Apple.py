import pygame
import random

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 640
GRID_SIZE = 16

class Apple:
    def __init__(self):
        self.position = self.generate_position()

    def generate_position(self):
        x = random.randint(0, (SCREEN_WIDTH - GRID_SIZE) // GRID_SIZE) * GRID_SIZE
        y = random.randint(0, (SCREEN_HEIGHT - GRID_SIZE) // GRID_SIZE) * GRID_SIZE
        return pygame.Rect(x, y, GRID_SIZE, GRID_SIZE)

    def detect_collision(self, snake_head):
        if self.position.colliderect(snake_head):
            self.position = self.generate_position()
            return True
        return False

    def draw(self, screen):
        pygame.draw.rect(screen, (255, 0, 0), self.position)

