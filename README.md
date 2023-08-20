# AI Self-Driving Car

<img src="Map%20and%20Graph/Path-3.png" width="600" height="600">

**AI Self-Driving Car** is an innovative project that explores the world of autonomous vehicles using deep reinforcement learning. Experience the future of transportation with cutting-edge AI technology.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)

## Table of Contents
- [Introduction](#introduction)
- [Neural Network Architecture](#neural-network-architecture)
- [Experience Replay](#experience-replay)
- [Deep Q Learning](#deep-q-learning)
- [Installation and Usage](#installation-and-usage)
- [Results and Insights](#results-and-insights)
- [Contributing](#contributing)

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)

## Introduction

The **AI Self-Driving Car** project focuses on creating an autonomous vehicle using deep reinforcement learning techniques. The project aims to simulate a self-driving car's navigation in various environments and scenarios, demonstrating how AI agents can learn to make driving decisions.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)

## Neural Network Architecture

The neural network architecture utilized in this project plays a crucial role in enabling the self-driving car to learn and adapt. The architecture consists of several layers, including fully connected layers, that help the AI agent process sensory input and make driving decisions.

[Link to AI Code](ai.py)

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)

## Experience Replay

Experience replay is a fundamental concept in reinforcement learning that enhances the AI's learning process. By storing and randomly sampling past experiences, the AI agent can learn from diverse situations and improve decision-making over time.

[Link to Experience Replay Code](ai.py)

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)

## Deep Q Learning

Deep Q Learning forms the core of the AI's decision-making process. This reinforcement learning algorithm helps the self-driving car learn optimal actions through interactions with the environment. The AI agent gradually improves its driving skills by maximizing cumulative rewards.

[Link to Deep Q Learning Code](ai.py)

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)

## Installation and Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/AI-Self-Driving-Car.git
    cd AI-Self-Driving-Car
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the self-driving car simulation:
     ```bash
    python map.py
    ```

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)

## Results and Insights
Gain insights into the AI Self-Driving Car's performance and progression over time. Below are visualizations of the AI agent's scores and statistics in different maps:

### Map 1
<img src="Map%20and%20Graph/Path-1.png" width="600" height="600">
### Stats:
<img src="Map%20and%20Graph/Graph-3.png" width="400" height="400">

### Map 2
<img src="Map%20and%20Graph/Path-2.png" width="600" height="600">
### Stats:
<img src="Map%20and%20Graph/Graph-2.png" width="400" height="400">

### Map 3
<img src="Map%20and%20Graph/Path-3.png" width="600" height="600">
### Stats:
<img src="Map%20and%20Graph/Graph-3.png" width="400" height="400">

Current Task: The car's initial task is to reach the top-left corner of the map. Once achieved, the destination switches automatically to the bottom-right corner. The cycle repeats, alternating between top-left and bottom-right.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)

## Contributing
Contribute to this exciting project by submitting pull requests, suggesting improvements, or sharing your insights. Your contributions help advance the field of AI and autonomous vehicles.

Disclaimer: This project is for educational and experimental purposes. The AI model's performance in the simulation does not reflect real-world driving scenarios.
