Environment Design:
The GridWorld is based on a fixed map of size 15x10 containing walls and free paths. At the beginning of each episode:
Harry(blue circle), the Cup(yellow circle), and the Death Eater(red circle) are randomly placed in free cells(obstacle free obviously)
The Death Eater follows Harry using Breadth-First Search(BFS).
Harry navigates using a discrete action space: up, down, left, right.

Since both Harry’s and the Death Eater’s positions influence decision-making, the Q-table is indexed using both.
Q[harry_state,death_state,action] (a 3-D array)
where, harry_state and death_state are flattened indices of Harry’s and Death Eater’s positions on the grid.
I thought of including the cup's position too in the Q-table but in that case the run time was very high so I excluded it. 

Reward Design:
The reward function is shaped to:
Encourage goal-reaching: +75 if Harry wins the Cup
Discourage capture: -75 if Harry is caught by the Death Eater
Encourage progress toward the goal: +10 if Harry moves closer to the Cup
Penalize proximity to the Death Eater: -10 if Harry moves closer to the Death Eater
Step penalty: -1 per step
Episode timeout: Episode ends with zero reward after 200 steps. This was done because when I was visualizaing the pygame, I saw Harry oscillating several times and not moving to any new cells.

Hyperparameters:
The following hyperparameters were chosen and tuned:
Episodes: 20000. A large number of episodes was used to ensure sufficient exploration and convergence of the policy. Even after updating the Q values for such large number of episodes the success-rate is very poor. 
Learning Rate(alpha): alpha=max(0.1,0.5*(1-ep/episodes))
Decaying learning rate encourages rapid learning in early episodes and stabilization later. It starts at 0.5 and linearly decreases to 0.1. When the learning rate was constant the success rate was even less than 4 percent so I had to switch to this method.
Discount Factor(gamma): 0.9
A high discount factor emphasizes long-term rewards, encouraging Harry to reach the Cup rather than just staying away from the Death Eater.
Exploration Rate: epsilon=max(0.05,1-ep/M) where M=1000
This epsilon-greedy strategy ensures sufficient exploration early on, then transitions toward exploitation over time.

Q-table Dimensions:
Since the grid has 150 positions and 4 possible actions: This accommodates all combinations of Harry's and Death Eater's positions.

Performance Metrics:
I have uploaded the links of the three plots. 
There was not even a single instance where Harry won the cup 10 times consecutively. 
The success rate of the model is almost 5.5% per 100 episode.

Final Testing:
After training, the Q-table is saved as q_table.npy. A separate evaluation script loads this Q-table and runs the environment with rendering to showcase the trained agent's behavior. No learning occurs during this phase; the policy purely exploits the learned Q-values.

These are the three plots which were displayed when I previously ran 20,000 episodes.
Plot 1:Rewards per-episode
![image](https://github.com/user-attachments/assets/7eb38691-1ecf-416e-a62a-dc7657f7bfed)
Plot 2:Success Rate per 100 episodes
![image](https://github.com/user-attachments/assets/615c3f6b-69f4-4a21-b9a7-74258d15ab28)
Plot 3:Moving Average Rewards over 100 episodes
![image](https://github.com/user-attachments/assets/1d9c2e71-dc8f-476d-87d8-b3c1f0745cb8)
