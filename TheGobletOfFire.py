import pygame
import numpy as np
import gymnasium as gym
from gym import spaces
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt

MAPS={"15x10":[ "111111111111111",
                "100000000000001",
                "101101010110001",
                "100100000010001",
                "100001001000001",
                "101001101011101",
                "101001001000101",
                "101100011100001",
                "100000000000001",
                "111111111111111"]}
class GridWorldEnvironment(gym.Env):
    metadata={"render modes":["human","rgb_array"],"render_fps": 4}
    FREE=0
    OBSTACLE=-1
    MOVES={0:(-1,0),1:(1,0),2:(0,-1),3:(0,1)}
    CELL_SIZE=40  
    COLOR_FREE=(255,255,255)
    COLOR_OBSTACLE=(0,0,0)
    COLOR_HARRY=(0,0,255)
    COLOR_CUP=(255,215,0)
    COLOR_DEATH_EATER=(255,0,0)

    def parse_obstacle_map(self,obstacle_map):
        if isinstance(obstacle_map,str):
            obstacle_map=MAPS[obstacle_map]
        grid=np.array([[self.FREE if char=='0' else self.OBSTACLE for char in row]
        for row in obstacle_map])
        return grid

    def __init__(self,obstacle_map:str|list[str],render_mode:str|None=None):
        self.render_mode=render_mode
        self.obstacles=self.parse_obstacle_map(obstacle_map)
        self.nrow, self.ncol=self.obstacles.shape
        self.action_space=spaces.Discrete(len(self.MOVES))
        self.observation_space=spaces.Discrete(n=self.nrow*self.ncol)
        self.fig=None
        self.fps=self.metadata['render_fps']
        self.screen=None
        self.clock=None

        if self.render_mode=="human":
            pygame.init()
            self.screen= pygame.display.set_mode((self.ncol*self.CELL_SIZE,self.nrow*self.CELL_SIZE))
            pygame.display.set_caption("GridWorld")
            self.clock=pygame.time.Clock()

    def render(self):
        if self.render_mode!="human":
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        self.screen.fill((255,255,255))

        for row in range(self.nrow):
            for col in range(self.ncol):
                rect = pygame.Rect(col*self.CELL_SIZE,row*self.CELL_SIZE,self.CELL_SIZE,self.CELL_SIZE)
                if self.obstacles[row,col]==self.OBSTACLE:
                    color=self.COLOR_OBSTACLE
                else:
                    color=self.COLOR_FREE
                pygame.draw.rect(self.screen,color,rect)

                pygame.draw.rect(self.screen,(200,200,200),rect,1)  

        
        pygame.draw.circle(self.screen,self.COLOR_HARRY,
                        (self.harry[1]*self.CELL_SIZE+self.CELL_SIZE//2,
                            self.harry[0]*self.CELL_SIZE+self.CELL_SIZE//2),self.CELL_SIZE//3)

        pygame.draw.circle(self.screen,self.COLOR_CUP,
                        (self.cup[1]*self.CELL_SIZE+self.CELL_SIZE//2,
                            self.cup[0]*self.CELL_SIZE+self.CELL_SIZE//2),self.CELL_SIZE//3)

        pygame.draw.circle(self.screen,self.COLOR_DEATH_EATER,
                        (self.death_eater[1]*self.CELL_SIZE+self.CELL_SIZE//2,
                            self.death_eater[0]*self.CELL_SIZE+self.CELL_SIZE//2),self.CELL_SIZE//3)

        pygame.display.flip()
        self.clock.tick(self.fps)

    def reset(self):
        free_pos=np.argwhere(self.obstacles==self.FREE)
        idx=np.random.choice(len(free_pos),size=3,replace=False)
        self.harry=tuple(free_pos[idx[0]])
        self.cup=tuple(free_pos[idx[1]])
        self.death_eater=tuple(free_pos[idx[2]])
        return self.harry,self.cup,self.death_eater
    
    def bfs(self,death_eater,harry):
        queue=deque()
        queue.append((death_eater,[]))
        visited=[death_eater]

        while queue:
            curr,path=queue.popleft()
            if curr==harry:
                return path
            for dr,dc in self.MOVES.values():
                nr,nc=curr[0]+dr,curr[1]+dc
                adj=(nr,nc)
                pos=self.obstacles[nr,nc]
                if (0<=nr<self.nrow and 0<=nc<self.ncol and pos!=self.OBSTACLE and adj!=self.cup and adj not in visited):
                    visited.append(adj)
                    queue.append((adj,path+[adj]))
        
        return []

    def traverse(self,action):
        row,col=self.harry
        drow,dcol=self.MOVES[action]
        new_row,new_col=row+drow,col+dcol
        if (0<=new_row<self.nrow) and (0<=new_col<self.ncol):
            if self.obstacles[new_row,new_col]!=self.OBSTACLE:
                self.harry=(new_row,new_col) 
        path=self.bfs(self.death_eater,self.harry)
        if len(path)>=1:
            self.death_eater=path[0]
            
        return self.harry,self.death_eater
    
def init_q_table(n_states,n_actions):
    return np.zeros((n_states,n_states,n_actions))

def q_update(Q,s_harry,s_death,a,r,s_next_harry,s_next_death,alpha,gamma):
    Q[s_harry,s_death,a]=Q[s_harry,s_death,a]+alpha*(r+gamma*(np.max(Q[s_next_harry,s_next_death])-Q[s_harry,s_death,a]))
    return Q

def rewardfun(nrharry,ncharry,rh,ch,rd,cd,r_cup,c_cup,done,steps):
    reward=-1
    prevharrydeath=np.linalg.norm(np.array([rh,ch])-np.array([rd,cd]))
    newharrydeath=np.linalg.norm(np.array([nrharry,ncharry])-np.array([rd,cd]))
    prevharrycup=np.linalg.norm(np.array([rh,ch])-np.array([r_cup,c_cup]))
    newharrycup=np.linalg.norm(np.array([nrharry,ncharry])-np.array([r_cup,c_cup]))
    
    if (nrharry,ncharry)==(r_cup,c_cup):
        reward=100
        done=True
        return reward,done
    
    if(nrharry,ncharry)==(rd,cd):
        reward=-100
        done=True
        return reward,done
    
    if newharrydeath<prevharrydeath:
        reward=reward-25
    
    if newharrycup<prevharrycup:
        reward=reward+25

    if steps>=200:
        reward=0
        done=True
        return reward,done
    
    return reward,done 


reward_list=[]
episodes_num=20000
alpha=0.1
gamma=0.9
env=GridWorldEnvironment("15x10",render_mode="human")
Q=init_q_table(150,4)
count=-1
win=0
eps_max=1
eps_min=0.05
M=1000
capture=0
success_window=deque(maxlen=100)
success_rate=[]
for ep in range(episodes_num):
    alpha=max(0.1,0.5*(1-ep/episodes_num))
    epsilon=max(eps_min,eps_max-ep/M)
    harry,cup,death_eater=env.reset()
    r_harry,c_harry=harry
    r_cup,c_cup=cup
    r_death,c_death=death_eater
    done=False
    temp=0
    steps=0
    while not done:
        env.render()
        steps=steps+1
        state_harry=r_harry*env.ncol+c_harry
        state_death=r_death*env.ncol+c_death
        if np.random.random()<epsilon:
            action=np.random.choice(len(Q[state_harry,state_death]))
        else:
            action=np.argmax(Q[state_harry,state_death])
        harrypos,deathpos=env.traverse(action)
        rh,ch=harrypos
        rd,cd=deathpos
        reward,done=rewardfun(r_harry,c_harry,rh,ch,rd,cd,r_cup,c_cup,done,steps)
        r_harry,c_harry=rh,ch
        next_harry=r_harry*env.ncol+c_harry
        next_death=rd*env.ncol+cd
        Q=q_update(Q,state_harry,state_death,action,reward,next_harry,next_death,alpha,gamma)
        temp+=reward
    reward_list.append(temp)
    if reward==75:
        win=win+1
        capture=capture+1
        success_window.append(1)
    else:
        win=0
        success_window.append(0)

    if win==10 and count==-1:
        count=ep+1

    if (ep+1)%100==0:
        success_rate_n=sum(success_window)
        success_rate.append(success_rate_n)

    if (ep+1)%5000==0:
        print(f"Episode {ep+1} rewards:",temp)

print("Number of generations:",count)
print("Success rate: ",(capture/episodes_num)*100)

np.save("q_table.npy",Q)
#Plot 1: Per-episode reward
plt.plot(reward_list, label="Reward")
plt.title("Per-Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()
plt.show(block=True)

# Plot 2: Success rate per 100 episodes
plt.plot(range(100,len(success_rate)*100+1,100),success_rate,label="Success Rate")
plt.title("Success Rate per 100 Episodes")
plt.xlabel("Episode")
plt.ylabel("Success %")
plt.grid(True)
plt.legend()
plt.show(block=true)

#calculating the moving average over a window of 100 episodes
window=int(20000/100)
data=pd.Series(reward_list)
moving_avg=data.rolling(window=window).mean()
plt.plot(moving_avg)
plt.title("Moving Average Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show(block=True)
pygame.quit()   


