from Snake import Snake
from Apple import Apple
from Network import Agent
import pygame
import torch
from train import get_state,SCREEN_HEIGHT,SCREEN_WIDTH

FPS = 30

def load_agent(agent):
    agent.model.load_state_dict(torch.load("best_agent_model.pth"))
    agent.epsilon = 0.05
    agent.train_start = 0
    agent.model.eval()  

def main():
    
    pygame.init()
    font_name = None
    font_size = 24

    font = pygame.font.SysFont(font_name, font_size)

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    snake = Snake()
    apple = Apple()
    agent = Agent(state_dim=14, action_dim=4)
    load_agent(agent)

    reward = 0
    high_score = 0
    num_apples = 0

    run = True
    episode_rewards = []
    episode_lengths = []
    current_episode_length = 0
    time_elapsed_since_apple= 0    
    
    while run:
        screen.fill((0, 0, 0))
        apple.draw(screen)
        snake.draw(screen)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        state = get_state(snake, apple)
        action = agent.act(state,train=False)

        directions = [pygame.Vector2(1, 0), pygame.Vector2(-1, 0), pygame.Vector2(0, 1), pygame.Vector2(0, -1)]
        snake.change_direction(directions[action])

        snake.check_borders()
        consumed_apple = apple.detect_collision(snake.body[0])
        if consumed_apple> 0:
            snake.length += 1  
            num_apples += 1
            reward += 10 + (num_apples * 4) 
            time_elapsed_since_apple = 0
            
        else:
            distance_to_apple = abs(snake.body[0].x - apple.position.x) + abs(snake.body[0].y - apple.position.y)
            K = 0.00005
            time_elapsed_since_apple += 1
            reward -= K  * distance_to_apple + time_elapsed_since_apple * 0.000006

        snake.move()
        
        collision_penalty = snake.collision_detection()

        if collision_penalty:
            time_elapsed_since_apple = 0
            reward = 0
            snake.reset()
            apple = Apple()
            episode_rewards.append(num_apples)
            episode_lengths.append(current_episode_length)
            current_episode_length = 0
            num_apples = 0
            snake.length =  5

        else:
            current_episode_length += 1
        
        if num_apples > high_score:
            high_score = num_apples

        text_surface2 = font.render(f'Score: {num_apples}', True, (255, 255, 255))
        text_surface3 = font.render(f'High Score: {high_score}', True, (255, 255, 255))

        screen.blit(text_surface2, (10, 10))
        screen.blit(text_surface3, (10,30))

        pygame.display.update()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
