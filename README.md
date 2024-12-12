# Machine Learning 2 (CS 4052U) Take Home Exam

**Chosen Topic:** Reinforcement Learning of Practical Problems

____________________

# Utilizing Q-Learning in Gymnasium: Step by Step

## Brief Gymnasium Overview
[Gymnasium](https://gymnasium.farama.org/) is a useful Python library based on [OpenAI's Gym library](https://github.com/openai/gym?tab=readme-ov-file). This library provides a variety of environments for training and testing reinforcement learning algorithms. You control an agent by choosing from different actions, influencing what state the agent ends up in next. Each environment in Gymnasium provides a non-zero reward (negative or positive) after each step and/or upon reaching the desired end goal. These rewards help reinforcement learning algorithms determine the effectiveness of actions taken in a given state, allowing adjustments in future episodes to determine the best possible action for each state. The environments offer various options that can increase the complexity of finding optimal solutions.

## The Frozen Lake Environment
For the purpose of testing Q-Learning in a relatively simple environment, I will be utilizing the [Frozen Lake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) environment. The goal of Frozen Lake is to cross a frozen lake from the starting point to the goal without falling into any holes. Frozen Lake comes in both a 4x4 and 8x8 environment, for this guide I will be using the 8x8 environment to make the challenge slightly more complex, and gather more insights in the process.
![Frozen Lake 8x8 Environment](Resources/8x8preview.gif)

### Understanding and Traversing the Environment
||||
|------------|---------|------------|
|Action Space|Discrete(4)|0: Move left, 1: Move down, 2: Move right, 3: Move up|
|Observation Space|Discrete(64)|0-63 (Position calculation: current_row *ncols + current_col)

![Observation and Action Space](Resources/ObserveAction.png)
</br>
In the above image we can see that the observation space is represented as integers from 0-63 where 0 is the starting position (top left) and 63 is the goal position (bottom right). You can calculate a position as the `current_row * ncols + current_col`. As mentioned in the table, the agent can take 1 of 4 actions:
- 0: Move left
- 1: Move down
- 2: Move right
- 3: Move up
  
In this example, taking action 1 (move down) moves the agent to position 17 (unless `is_slippery` is enabled, while is explained below).


#### is_slippery
When creating the environment, `is_slippery` can be set to True or False. If disabled, you will always move in your intended direction. If enabled, you will only move in your intended direction 1/3 of the time, otherwise, you have an equal 1/3 chance of slipping in either perpendicular direction. For example:
</br>
is_slippery=False             |  is_slippery=True
:-------------------------:|:-------------------------:
![](Resources/notslippery_chances.png)  |  ![](resources/slippery_chances.png)

When taking action 2 (go right), the `is_slippery` setting determines movement probabilities. If disabled, you always move right. If enabled, you move right with only a 1/3 probability, and there's an equal chance of moving in either perpendicular direction. In this case you could go up or down as well. 
<br/><br/>
**Enabling is_slippery increases complexity. Instead of simply finding the shortest path to the goal, you must now find the path that maximizes your chances of reaching the goal without falling into a hole.** In some other environments, a shorter but riskier path might be preferable, if longer paths incur negative rewards / penalties. However, as we'll see, Frozen Lake uses a simple reward structure.

#### Rewards and Endings
The rewards structure for Frozen Lake is very simple. You are only rewarded for eventually reaching the goal. There is no penalty for taking too many actions, except for a time limit.

|Reached|Reward|
|---------|-----------|
|Goal|+1|
|Hole|0|
|Frozen|0|

**Possible Endings:**
- Goal Reached
- Player moves into a hole
- Length of episode (actions taken) reach 200 (100 if using 4x4 environment)

While there is technically no negative reward, reaching the episode limit or falling into a hole is undesirable as we will not get a reward for that episode, and we will be unable to tell if the actions we took in that episode were in the "right direction" up to a certain point as we receive no positive feedback for partial progress.

## Setup
### Initializing the Environment

## Q-Learning Implementation
### Results


## Full Code

## Resources Used
- AI (CSCI 4610U) Lectures: Winter 2024 Semester
- ML2 (CSCI 4052U) Lectures: Fall 2024 Semester
- https://www.youtube.com/watch?v=ZhoIgo3qqLU - FrozenLake Gymnasium
- https://gymnasium.farama.org/environments/toy_text/frozen_lake/