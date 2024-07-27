# DQN Agent for Snake Game using Pygame and PyTorch

This repository hosts a Deep Q-Network (DQN) agent designed to play the classic Snake game. The project leverages PyTorch for the deep learning component and Pygame for game rendering. This initiative served as my introduction to the fascinating world of reinforcement learning.

## Features

- **Deep Q-Network (DQN) Implementation**: The agent learns to play Snake through a DQN algorithm, optimizing its actions based on game states.
- **Pygame Integration**: Provides a visual interface for the game, enhancing the learning experience.
- **Comprehensive Training and Inference**: Includes scripts for training the agent from scratch and for running the trained agent to play the game.
- **Hyperparameter Customization**: Easily adjustable settings to experiment with different training configurations.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/dqn-snake-game.git
    cd dqn-snake-game
    ```

2. Install the required dependencies:
   Pytorch,pygame,numpy (and matplotlib if you're going to be training)
## Usage

### Training the Agent

Initiate the training process with:
```bash
python train.py
```
This command starts training the DQN agent, where it will iteratively improve its performance through gameplay.

### Running the Trained Agent

To see the trained agent in action, execute:
```bash
python play.py
```
This will run the trained agent, allowing it to play the Snake game autonomously, this agent was trained for close to 10 hours.

The agent struggles a little in the early game but does quite well in the late game, if you find it to be slow and want to speed it up so that it gets to late game quicker just modify the FPS variable in play.py and set it to 60

https://github.com/user-attachments/assets/22b83dd0-6eb0-4b0f-86b7-89301a3bf7b2


## Repository Structure

- `train.py`: Script to train the DQN agent.
- `play.py`: Script to run the trained agent.
- `dqn.py`: Core implementation of the DQN algorithm.
- `snake_game.py`: Pygame-based implementation of the Snake game.
- `model.py`: Neural network model definition used by the DQN agent.
- `utils.py`: Utility functions for various tasks.
- `requirements.txt`: List of dependencies required to run the project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.



