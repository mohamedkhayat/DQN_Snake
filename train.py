from Snake import Snake
from Apple import Apple
from Network import Agent
import numpy as np
import pygame
import matplotlib.pyplot as plt
import random
import torch

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 640
FPS = 120


def get_state(snake,apple):
    head = snake.body[0]
    tail = snake.body[-1]

    head_x_norm = head.x / SCREEN_WIDTH
    head_y_norm = head.y / SCREEN_HEIGHT
    tail_x_norm = tail.x / SCREEN_WIDTH
    tail_y_norm = tail.y / SCREEN_HEIGHT

    manhattan_dist = abs(head.x - apple.position.x) + abs(head.y - apple.position.y)
    #add time since last apple
    direction_to_apple = [
        (apple.position.x - head.x) / SCREEN_WIDTH,
        (apple.position.y - head.y) / SCREEN_HEIGHT
    ]

    left = int(apple.position.x < head.x)
    right = int(apple.position.x > head.x)
    up = int(apple.position.y < head.y)
    down = int(apple.position.y > head.y)

    state = [
        head_x_norm, head_y_norm,
        tail_x_norm, tail_y_norm,
        snake.length,
        snake.direction.x, snake.direction.y,
        manhattan_dist,
        *direction_to_apple,
        *[left, right, up, down] ]

    return np.array(state)

def save_agent(agent, filename='agent'):
    torch.save(agent.model.state_dict(),filename+"_model.pth")

def main():
    
    pygame.init()

    font_name = 'dejavusansbold'
    font_size = 24
    font = pygame.font.SysFont(font_name, font_size)

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    snake = Snake()
    apple = Apple()
    agent = Agent(state_dim=14, action_dim=4)

    run = True
    latest_reward = 0
    reward = 0
    high_score = 0
    num_apples = 0
    highest_reward = 2000
    
    episode_rewards = []
    episode_lengths = []
    
    current_episode_reward = 0
    current_episode_length = 0
    time_elapsed_since_apple= 0    

    random_length_phase = False  
    length_randomization_interval = 100
    while run:
        screen.fill((0, 0, 0))
        apple.draw(screen)
        snake.draw(screen)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:  
                    save_agent(agent, filename='manual_agent')

        state = get_state(snake, apple)
        action = agent.act(state)

        directions = [pygame.Vector2(1, 0), pygame.Vector2(-1, 0), pygame.Vector2(0, 1), pygame.Vector2(0, -1)]
        snake.change_direction(directions[action])

        snake.check_borders()
        consumed_apple = apple.detect_collision(snake.body[0])
        
        if consumed_apple:
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
        #tweak time
        if collision_penalty or time_elapsed_since_apple/60 >= 30:
            time_elapsed_since_apple = 0
            latest_reward = reward
            reward = -10
            done = True
            snake.reset()
            apple = Apple()
            reward = 0
            episode_rewards.append(num_apples)
            episode_lengths.append(current_episode_length)
            current_episode_reward = 0
            current_episode_length = 0
            num_apples = 0

            if random_length_phase:
                snake.length = random.randint(5, 20)  
            else:
                snake.length =  5

        else:
            done = False
            current_episode_reward += reward
            current_episode_length += 1

        next_state = get_state(snake, apple)
    
        if high_score < num_apples:
            high_score = num_apples
         
        text_surface = font.render(f'Reward: {reward:.2f}', True, (255, 255, 255))
        text_surface2 = font.render(f'Score: {num_apples}', True, (255, 255, 255))
        text_surface3 = font.render(f'High Score: {high_score}', True, (255, 255, 255))
        text_surface4 = font.render(f'Time w/o apple: {time_elapsed_since_apple/FPS:.2f}', True, (255, 255, 255))

        screen.blit(text_surface, (10, 10))
        screen.blit(text_surface2, (10, 30))
        screen.blit(text_surface3, (10, 50))
        screen.blit(text_surface4, (SCREEN_WIDTH - 250, 10))

        agent.remember(state, action, reward, next_state, done)
        agent.replay()

        if random_length_phase and (len(episode_rewards) >= length_randomization_interval):
            print("end random_length phase")
            random_length_phase = False
           
        if latest_reward > highest_reward:
                highest_reward = latest_reward
                print("saved model")
                save_agent(agent, filename='best_agent')

        pygame.display.update()
        clock.tick(FPS)

    pygame.quit()

    episodes = list(range(len(episode_rewards)))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(episodes, episode_rewards, label='Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(episodes, episode_lengths, label='Episode Length', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.title('Episode Length per Episode')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()
